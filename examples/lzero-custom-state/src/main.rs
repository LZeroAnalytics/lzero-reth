use alloy_consensus::{Header, Transaction as _};
use alloy_eips::{eip6110, eip7685::Requests};
use alloy_primitives::U256;
use alloy_primitives::Bytes;
use clap::Parser;
use reth::{
    api::{FullNodeTypes, NodePrimitives, NodeTypesWithEngine},
    builder::{
        components::{
            ExecutorBuilder,
            PayloadServiceBuilder,
        },
        PayloadBuilderConfig,
        rpc::EngineValidatorBuilder,
        BuilderContext,
        NodeHandle,
    },
    chainspec::{ChainSpec, EthereumChainSpecParser, EthereumHardfork, EthereumHardforks},
    cli::Cli,
    consensus::ConsensusError,
    core::primitives::SignedTransaction,
    payload::{PayloadBuilderHandle, PayloadBuilderService, EthBuiltPayload, EthPayloadBuilderAttributes},
    providers::{ProviderError, CanonStateSubscriptions},
    revm::{
        interpreter::primitives::AccountInfo,
        primitives::{hex, Bytecode, EnvWithHandlerCfg, ResultAndState},
        Database, DatabaseCommit, State,
    },
    rpc::{api::eth::helpers::Call, types::TransactionTrait},
    transaction_pool::{PoolTransaction, TransactionPool},
};
use reth_basic_payload_builder::{PayloadBuilder, BuildArguments, BuildOutcome, PayloadConfig, BasicPayloadJobGenerator};
use reth_ethereum_payload_builder::{EthereumBuilderConfig, EthereumPayloadBuilder};
use reth_evm::{
    env::EvmEnv,
    execute::{
        balance_increment_state, BlockExecutionError, BlockExecutionStrategy,
        BlockExecutionStrategyFactory, BlockValidationError, ExecuteOutput,
    },
    state_change::post_block_balance_increments,
    system_calls::SystemCaller,
    ConfigureEvm, ConfigureEvmEnv, TxEnvOverrides,
};
use reth_evm_ethereum::{
    dao_fork::{DAO_HARDFORK_ACCOUNTS, DAO_HARDFORK_BENEFICIARY},
    eip6110::parse_deposits_from_receipts,
    EthEvmConfig,
};
use reth_node_ethereum::{node::EthereumAddOns, BasicBlockExecutorProvider, EthEngineTypes, EthereumNode};
use reth_primitives::{BlockWithSenders, EthPrimitives, Receipt, TransactionSigned};
use std::{
    fmt::Display,
    hash::Hash,
    sync::Arc,
};
use reth::revm::primitives::{BlockEnv, CfgEnvWithHandlerCfg};
use reth::version::default_extra_data_bytes;
use reth_payload_builder::{PayloadBuilderError};
use reth_basic_payload_builder::{BasicPayloadJobGeneratorConfig};
use reth_chainspec::ChainSpecProvider;
use reth_revm::database::StateProviderDatabase;
use reth_storage_api::StateProviderFactory;

fn main() {
    Cli::<EthereumChainSpecParser>::parse()
        .run(|builder, args| async move {
            let handle = builder
                .with_types::<EthereumNode>()
                .with_components(
                    EthereumNode::components()
                        .executor(CustomExecutorBuilder::default())
                        .payload(CustomPayloadServiceBuilder::default())
                )
                .with_add_ons(EthereumAddOns::default())
                .launch()
                .await?;

            // launch the node
            let NodeHandle { node, node_exit_future } = handle;

            node_exit_future.await
        })
        .unwrap();
}

/// A custom executor builder.
#[derive(Debug, Default, Clone, Copy)]
pub struct CustomExecutorBuilder;

impl<Types, Node> ExecutorBuilder<Node> for CustomExecutorBuilder
where
    Types: NodeTypesWithEngine<ChainSpec = ChainSpec, Primitives = EthPrimitives>,
    Node: FullNodeTypes<Types = Types>,
{
    type EVM = EthEvmConfig;
    type Executor = BasicBlockExecutorProvider<CustomExecutorStrategyFactory>;

    async fn build_evm(
        self,
        ctx: &BuilderContext<Node>,
    ) -> eyre::Result<(Self::EVM, Self::Executor)> {
        let chain_spec = ctx.chain_spec();
        let evm_config = EthEvmConfig::new(ctx.chain_spec());
        let strategy_factory = CustomExecutorStrategyFactory {
            chain_spec: chain_spec.clone(),
            evm_config: evm_config.clone(),
        };
        let executor = BasicBlockExecutorProvider::new(strategy_factory);

        Ok((evm_config, executor))
    }
}

#[derive(Clone)]
pub struct CustomExecutorStrategyFactory<EvmConfig = EthEvmConfig> {
    chain_spec: Arc<ChainSpec>,
    evm_config: EvmConfig,
}

impl <EvmConfig>BlockExecutionStrategyFactory for CustomExecutorStrategyFactory<EvmConfig>
where
    EvmConfig: Clone
    + Unpin
    + Sync
    + Send
    + 'static
    + ConfigureEvm<
        Header = alloy_consensus::Header,
        Transaction = reth_primitives::TransactionSigned,
    >,
{
    type Primitives = EthPrimitives;

    type Strategy<DB: Database<Error: Into<ProviderError> + Display>> =
    CustomExecutorStrategy<DB, EvmConfig>;

    fn create_strategy<DB>(&self, db: DB) -> Self::Strategy<DB>
    where
        DB: Database<Error: Into<ProviderError> + Display>,
    {
        let state =
            State::builder().with_database(db).with_bundle_update().without_state_clear().build();
        CustomExecutorStrategy::new(state, self.chain_spec.clone(), self.evm_config.clone())
    }
}

#[allow(missing_debug_implementations)]
pub struct CustomExecutorStrategy<DB, EvmConfig>
where
    EvmConfig: Clone,
{
    /// The chainspec
    chain_spec: Arc<ChainSpec>,
    /// How to create an EVM.
    evm_config: EvmConfig,
    /// Optional overrides for the transactions environment.
    tx_env_overrides: Option<Box<dyn TxEnvOverrides>>,
    /// Current state for block execution.
    state: State<DB>,
    /// Utility to call system smart contracts.
    system_caller: SystemCaller<EvmConfig, ChainSpec>,
}

impl<DB, EvmConfig> CustomExecutorStrategy<DB, EvmConfig>
where
    EvmConfig: Clone,
{
    /// Creates a new [`CustomExecutorStrategy`]
    pub fn new(state: State<DB>, chain_spec: Arc<ChainSpec>, evm_config: EvmConfig) -> Self {
        let system_caller = SystemCaller::new(evm_config.clone(), chain_spec.clone());
        Self { state, chain_spec, evm_config, system_caller, tx_env_overrides: None }
    }
}

impl<DB, EvmConfig> CustomExecutorStrategy<DB, EvmConfig>
where
    DB: Database<Error: Into<ProviderError> + Display>,
    EvmConfig: ConfigureEvm<Header = alloy_consensus::Header>,
{
    /// Configures a new evm configuration and block environment for the given block.
    ///
    /// # Caution
    ///
    /// This does not initialize the tx environment.
    fn evm_env_for_block(&self, header: &alloy_consensus::Header) -> EnvWithHandlerCfg {
        let EvmEnv { cfg_env_with_handler_cfg, block_env } =
            self.evm_config.cfg_and_block_env(header);
        EnvWithHandlerCfg::new_with_cfg_env(cfg_env_with_handler_cfg, block_env, Default::default())
    }
}

impl<DB, EvmConfig> BlockExecutionStrategy for CustomExecutorStrategy<DB, EvmConfig>
where
    DB: Database<Error: Into<ProviderError> + Display>,
    EvmConfig: ConfigureEvm<
        Header = alloy_consensus::Header,
        Transaction = reth_primitives::TransactionSigned,
    >,
{
    type DB = DB;
    type Primitives = EthPrimitives;
    type Error = BlockExecutionError;

    fn init(&mut self, tx_env_overrides: Box<dyn TxEnvOverrides>) {
        self.tx_env_overrides = Some(tx_env_overrides);
    }

    fn apply_pre_execution_changes(
        &mut self,
        block: &BlockWithSenders,
    ) -> Result<(), Self::Error> {

        // Set state clear flag if the block is after the Spurious Dragon hardfork.
        let state_clear_flag = (*self.chain_spec).is_spurious_dragon_active_at_block(block.header.number);
        self.state.set_state_clear_flag(state_clear_flag);

        // Initialize the EVM environment for the block.
        {
            let env = self.evm_env_for_block(&block.header);
            let mut evm = self.evm_config.evm_with_env(&mut self.state, env);

            // Handle system-level pre-execution changes if applicable.
            self.system_caller.apply_pre_execution_changes(&block.block, &mut evm)?;
        }

        Ok(())
    }

    fn execute_transactions(
        &mut self,
        block: &BlockWithSenders,
    ) -> Result<ExecuteOutput<Receipt>, Self::Error> {
        const CODE_STORAGE_GAS: u64 = 200;

        let env = self.evm_env_for_block(&block.header);
        let mut evm = self.evm_config.evm_with_env(&mut self.state, env);

        let mut cumulative_gas_used = 0;
        let mut receipts = Vec::with_capacity(block.body.transactions.len());

        for (sender, transaction) in block.transactions_with_sender() {
            // Check if bytecode needs to be injected before the transaction executes.
            if let Some(recipient) = transaction.to() {
                if !transaction.input().is_empty() {
                    // Temporarily release the mutable borrow on `self.state` to inject bytecode.
                    drop(evm); // Explicitly drop `evm` to release the mutable borrow.

                    let inject_bytecode = {
                        if let Some(account) = self.state.database.basic(recipient).unwrap_or_default() {
                            account.code.is_none()
                                || account
                                .code
                                .as_ref()
                                .map_or(true, |code| code.is_empty())
                        } else {
                            true
                        }
                    };

                    if inject_bytecode {
                        let custom_bytecode = hex::decode(
                            "6080604052348015600e575f5ffd5b5060405161020f38038061020f8339818101604052810190602e9190606b565b805f81905550506091565b5f5ffd5b5f819050919050565b604d81603d565b81146056575f5ffd5b50565b5f815190506065816046565b92915050565b5f60208284031215607d57607c6039565b5b5f6088848285016059565b91505092915050565b6101718061009e5f395ff3fe608060405234801561000f575f5ffd5b506004361061003f575f3560e01c8063209652551461004357806347f1aae714610061578063552410771461007f575b5f5ffd5b61004b61009b565b60405161005891906100c9565b60405180910390f35b6100696100a3565b60405161007691906100c9565b60405180910390f35b61009960048036038101906100949190610110565b6100a8565b005b5f5f54905090565b5f5481565b805f8190555050565b5f819050919050565b6100c3816100b1565b82525050565b5f6020820190506100dc5f8301846100ba565b92915050565b5f5ffd5b6100ef816100b1565b81146100f9575f5ffd5b50565b5f8135905061010a816100e6565b92915050565b5f60208284031215610125576101246100e2565b5b5f610132848285016100fc565b9150509291505056fea264697066735822122074dc3cb86974667f3422b096bc112cbe113575bbe68afa378e85c7e3b00448b264736f6c634300081c0033",
                        )
                            .unwrap();

                        let bytecode_size = custom_bytecode.len() as u64;

                        // Increment gas for bytecode storage.
                        let bytecode_gas_cost = CODE_STORAGE_GAS * bytecode_size;
                        cumulative_gas_used += bytecode_gas_cost;
                        self.state.insert_account(
                            recipient,
                            AccountInfo {
                                balance: U256::ZERO,
                                nonce: 0,
                                code_hash: reth_revm::primitives::keccak256(&custom_bytecode),
                                code: Some(Bytecode::new_raw(custom_bytecode.into())),
                            },
                        );
                        println!("Injected custom bytecode for address: {recipient:?}");
                    }

                    // Reinitialize the EVM with the updated state after bytecode injection.
                    let env = self.evm_env_for_block(&block.header);
                    evm = self.evm_config.evm_with_env(&mut self.state, env);
                }
            }

            // Continue with transaction execution.
            let block_available_gas = block.header.gas_limit - cumulative_gas_used;
            if transaction.gas_limit() > block_available_gas {
                return Err(BlockValidationError::TransactionGasLimitMoreThanAvailableBlockGas {
                    transaction_gas_limit: transaction.gas_limit(),
                    block_available_gas,
                }
                    .into());
            }

            self.evm_config.fill_tx_env(evm.tx_mut(), transaction, *sender);

            if let Some(tx_env_overrides) = &mut self.tx_env_overrides {
                tx_env_overrides.apply(evm.tx_mut());
            }

            // Execute the transaction.
            let result_and_state = evm.transact().map_err(move |err| {
                let new_err = err.map_db_err(|e| e.into());
                BlockValidationError::EVM {
                    hash: transaction.recalculate_hash(),
                    error: Box::new(new_err),
                }
            })?;

            self.system_caller.on_state(&result_and_state.state);

            let ResultAndState { result, state } = result_and_state;
            evm.db_mut().commit(state);

            cumulative_gas_used += result.gas_used();

            // Collect the transaction receipt.
            receipts.push(Receipt {
                tx_type: transaction.tx_type(),
                success: result.is_success(),
                cumulative_gas_used,
                logs: result.into_logs(),
                ..Default::default()
            });
        }

        Ok(ExecuteOutput {
            receipts,
            gas_used: cumulative_gas_used,
        })
    }

    fn apply_post_execution_changes(
        &mut self,
        block: &BlockWithSenders,
        receipts: &[Receipt],
    ) -> Result<Requests, Self::Error> {
        let env = self.evm_env_for_block(&block.header);
        let mut evm = self.evm_config.evm_with_env(&mut self.state, env);

        let requests = if self.chain_spec.is_prague_active_at_timestamp(block.timestamp) {
            // Collect all EIP-6110 deposits
            let deposit_requests = parse_deposits_from_receipts(&self.chain_spec, receipts)?;

            let mut requests = Requests::default();

            if !deposit_requests.is_empty() {
                requests.push_request_with_type(eip6110::DEPOSIT_REQUEST_TYPE, deposit_requests);
            }

            requests.extend(self.system_caller.apply_post_execution_changes(&mut evm)?);
            requests
        } else {
            Requests::default()
        };
        drop(evm);

        let mut balance_increments = post_block_balance_increments(&self.chain_spec, &block.block);

        // Irregular state change at Ethereum DAO hardfork
        if self.chain_spec.fork(EthereumHardfork::Dao).transitions_at_block(block.number) {
            // drain balances from hardcoded addresses.
            let drained_balance: u128 = self
                .state
                .drain_balances(DAO_HARDFORK_ACCOUNTS)
                .map_err(|_| BlockValidationError::IncrementBalanceFailed)?
                .into_iter()
                .sum();

            // return balance to DAO beneficiary.
            *balance_increments.entry(DAO_HARDFORK_BENEFICIARY).or_default() += drained_balance;
        }
        // increment balances
        self.state
            .increment_balances(balance_increments.clone())
            .map_err(|_| BlockValidationError::IncrementBalanceFailed)?;
        // call state hook with changes due to balance increments.
        let balance_state = balance_increment_state(&balance_increments, &mut self.state)?;
        self.system_caller.on_state(&balance_state);

        Ok(requests)
    }

    fn state_ref(&self) -> &State<Self::DB> {
        &self.state
    }

    fn state_mut(&mut self) -> &mut State<Self::DB> {
        &mut self.state
    }

    fn validate_block_post_execution(
        &self,
        block: &BlockWithSenders<<Self::Primitives as NodePrimitives>::Block>,
        _receipts: &[<Self::Primitives as NodePrimitives>::Receipt],
        _requests: &Requests,
    ) -> Result<(), ConsensusError> {
        Ok(())
    }
}

//
// Custom Payload
//
#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct CustomPayloadServiceBuilder;

impl<Node, Pool> PayloadServiceBuilder<Node, Pool> for CustomPayloadServiceBuilder
where
    Node: FullNodeTypes<
        Types: NodeTypesWithEngine<
            Engine = EthEngineTypes,
            ChainSpec = ChainSpec,
            Primitives = EthPrimitives,
        >,
    >,
    Pool: TransactionPool<Transaction: PoolTransaction<Consensus = TransactionSigned>>
    + Unpin
    + 'static,
{
    async fn spawn_payload_service(
        self,
        ctx: &BuilderContext<Node>,
        pool: Pool,
    ) -> eyre::Result<PayloadBuilderHandle<<Node::Types as NodeTypesWithEngine>::Engine>> {
        let payload_builder = CustomPayloadBuilder::default();
        let conf = ctx.payload_builder_config();

        let payload_job_config = BasicPayloadJobGeneratorConfig::default()
            .interval(conf.interval())
            .deadline(conf.deadline())
            .max_payload_tasks(conf.max_payload_tasks());

        let payload_generator = BasicPayloadJobGenerator::with_builder(
            ctx.provider().clone(),
            pool,
            ctx.task_executor().clone(),
            payload_job_config,
            payload_builder,
        );
        let (payload_service, payload_builder) =
            PayloadBuilderService::new(payload_generator, ctx.provider().canonical_state_stream());

        ctx.task_executor().spawn_critical("payload builder service", Box::pin(payload_service));

        Ok(payload_builder)
    }
}

#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct CustomPayloadBuilder;

impl<Pool, Client> PayloadBuilder<Pool, Client> for CustomPayloadBuilder
where
    Client: StateProviderFactory + ChainSpecProvider<ChainSpec = ChainSpec>,
    Pool: TransactionPool<Transaction: PoolTransaction<Consensus = TransactionSigned>>,
{
    type Attributes = EthPayloadBuilderAttributes;
    type BuiltPayload = EthBuiltPayload;

    fn try_build(
        &self,
        args: BuildArguments<Pool, Client, Self::Attributes, Self::BuiltPayload>,
    ) -> Result<BuildOutcome<Self::BuiltPayload>, PayloadBuilderError> {
        let BuildArguments { client, pool, cached_reads, config, cancel, best_payload } = args;
        let PayloadConfig { parent_header, attributes } = config;

        let chain_spec = client.chain_spec();

        // This reuses the default EthereumPayloadBuilder to build the payload
        // but any custom logic can be implemented here
        reth_ethereum_payload_builder::EthereumPayloadBuilder::new(
            EthEvmConfig::new(chain_spec.clone()),
            EthereumBuilderConfig::new(default_extra_data_bytes()),
        )
            .try_build(BuildArguments {
                client,
                pool,
                cached_reads,
                config: PayloadConfig { parent_header, attributes },
                cancel,
                best_payload,
            })
    }

    fn build_empty_payload(
        &self,
        client: &Client,
        config: PayloadConfig<Self::Attributes>,
    ) -> Result<Self::BuiltPayload, PayloadBuilderError> {
        let PayloadConfig { parent_header, attributes } = config;
        let chain_spec = client.chain_spec();
        <reth_ethereum_payload_builder::EthereumPayloadBuilder as PayloadBuilder<Pool, Client>>::build_empty_payload(
            &reth_ethereum_payload_builder::EthereumPayloadBuilder::new(
                EthEvmConfig::new(chain_spec.clone()),
                EthereumBuilderConfig::new(default_extra_data_bytes())
            ),
            client,
            PayloadConfig { parent_header, attributes }
        )
    }
}