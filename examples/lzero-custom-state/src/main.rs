use alloy_consensus::{Header, Transaction as _, EMPTY_OMMER_ROOT_HASH};
use alloy_eips::{eip6110, eip7685::Requests, eip4844::MAX_DATA_GAS_PER_BLOCK, merge::BEACON_NONCE, eip7840::BlobParams};
use alloy_primitives::{Address, U256, B256};
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
        primitives::{hex, Bytecode, EnvWithHandlerCfg, ResultAndState,
                     BlockEnv, CfgEnvWithHandlerCfg, TxEnv, InvalidTransaction, EVMError},
        Database, DatabaseCommit, State,
        db::{states::bundle_state::BundleRetention}
    },
    rpc::{api::eth::helpers::Call, types::TransactionTrait},
    transaction_pool::{
        PoolTransaction,
        TransactionPool,
        BestTransactionsAttributes,
        error::InvalidPoolTransactionError,
        noop::NoopTransactionPool,
        BestTransactions,
        ValidPoolTransaction,
    },
    version::default_extra_data_bytes,
};
use reth_basic_payload_builder::{PayloadBuilder, BuildArguments, BuildOutcome, PayloadConfig, BasicPayloadJobGenerator, is_better_payload, commit_withdrawals};
use reth_chain_state::ExecutedBlock;
use reth_errors::RethError;
use reth_ethereum_payload_builder::{EthereumBuilderConfig, EthereumPayloadBuilder};
use reth_evm::{env::EvmEnv, execute::{
    balance_increment_state, BlockExecutionError, BlockExecutionStrategy,
    BlockExecutionStrategyFactory, BlockValidationError, ExecuteOutput,
}, state_change::post_block_balance_increments, system_calls::SystemCaller, ConfigureEvm, ConfigureEvmEnv, NextBlockEnvAttributes, TxEnvOverrides};
use reth_evm_ethereum::{
    dao_fork::{DAO_HARDFORK_ACCOUNTS, DAO_HARDFORK_BENEFICIARY},
    eip6110::parse_deposits_from_receipts,
    EthEvmConfig,
};
use reth_node_ethereum::{node::EthereumAddOns, BasicBlockExecutorProvider, EthEngineTypes, EthereumNode};
use reth_primitives::{BlockWithSenders, EthPrimitives, Receipt,
                      TransactionSigned, InvalidTransactionError,
                      proofs::{self},
                      Block, BlockBody, BlockExt,
};
use reth_execution_types::ExecutionOutcome;
use std::{
    fmt::Display,
    hash::Hash,
    sync::Arc,
};
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use alloy_primitives::map::HashMap;
use reth_payload_builder::{PayloadBuilderError};
use reth_basic_payload_builder::{BasicPayloadJobGeneratorConfig};
use reth_chainspec::{ChainSpecBuilder, ChainSpecProvider};
use reth_revm::database::StateProviderDatabase;
use reth_storage_api::StateProviderFactory;
use tracing::{debug, trace, warn};
use reth::api::{NodeTypesWithDBAdapter, PayloadBuilderAttributes};
use reth::blockchain_tree::noop::NoopBlockchainTree;
use reth::providers::ProviderFactory;
use reth::providers::providers::{BlockchainProvider, StaticFileProvider};
use reth::revm::db::{AccountStatus, BundleAccount, BundleState, CacheDB, EmptyDB, EmptyDBTyped, StorageWithOriginalValues};
use reth::revm::db::states::StorageSlot;
use reth::revm::{DatabaseRef, TransitionState};
use reth::rpc::eth::RpcNodeCore;
use reth_revm::primitives::Account;
use reth_db::{mdbx::DatabaseArguments, DatabaseEnv};

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
        Header = Header,
        Transaction = TransactionSigned,
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
    EvmConfig: ConfigureEvm<Header = Header>,
{
    /// Configures a new evm configuration and block environment for the given block.
    ///
    /// # Caution
    ///
    /// This does not initialize the tx environment.
    fn evm_env_for_block(&self, header: &Header) -> EnvWithHandlerCfg {
        let EvmEnv { cfg_env_with_handler_cfg, block_env } =
            self.evm_config.cfg_and_block_env(header);
        EnvWithHandlerCfg::new_with_cfg_env(cfg_env_with_handler_cfg, block_env, Default::default())
    }
}

impl<DB, EvmConfig> BlockExecutionStrategy for CustomExecutorStrategy<DB, EvmConfig>
where
    DB: Database<Error: Into<ProviderError> + Display>,
    EvmConfig: ConfigureEvm<
        Header = Header,
        Transaction = TransactionSigned,
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

    fn finish(&mut self) -> BundleState {
        if let Some(transition_state) = self.state.transition_state.clone() {
            for (address, transition_account) in &transition_state.transitions {
                if let Some(info) = &transition_account.info {
                    if let Some(code) = &info.code {
                        let code_hash = info.code_hash;
                        if !self.state.bundle_state.contracts.contains_key(&code_hash) {
                            self.state.bundle_state.contracts.insert(code_hash, code.clone());
                            let bundle_account = BundleAccount {
                                info: Some(info.clone()),
                                original_info: transition_account.previous_info.clone(),
                                storage: transition_account.storage.clone(),
                                status: transition_account.status.clone(),
                            };

                            self.state.bundle_state.state.insert(address.clone(), bundle_account);
                        }
                    }
                }
            }
        }
        self.state_mut().merge_transitions(BundleRetention::Reverts);
        self.state_mut().take_bundle()
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
        let evm_config = EthEvmConfig::new(chain_spec.clone());
        let builder_config = EthereumBuilderConfig::new(default_extra_data_bytes());

        let next_attributes = NextBlockEnvAttributes {
            timestamp: attributes.timestamp(),
            suggested_fee_recipient: attributes.suggested_fee_recipient(),
            prev_randao: attributes.prev_randao(),
            gas_limit: builder_config.gas_limit(parent_header.gas_limit),
        };

        let EvmEnv { cfg_env_with_handler_cfg, block_env } = evm_config
            .next_cfg_and_block_env(&parent_header, next_attributes)
            .map_err(PayloadBuilderError::other)?;

        let pool_for_closure = pool.clone();
        let best_transactions = move |attrs| pool_for_closure.best_transactions_with_attributes(attrs);

        lzero_custom_payload(
            evm_config,
            builder_config,
            BuildArguments {
                client,
                pool,
                cached_reads,
                config: PayloadConfig { parent_header, attributes },
                cancel,
                best_payload,
            },
            cfg_env_with_handler_cfg,
            block_env,
            best_transactions,
        )
    }

    fn build_empty_payload(
        &self,
        client: &Client,
        config: PayloadConfig<Self::Attributes>,
    ) -> Result<Self::BuiltPayload, PayloadBuilderError> {
        let PayloadConfig { parent_header, attributes } = config;
        let chain_spec = client.chain_spec();
        <EthereumPayloadBuilder as PayloadBuilder<Pool, Client>>::build_empty_payload(
            &EthereumPayloadBuilder::new(
                EthEvmConfig::new(chain_spec.clone()),
                EthereumBuilderConfig::new(default_extra_data_bytes())
            ),
            client,
            PayloadConfig { parent_header, attributes }
        )
    }
}

type BestTransactionsIter<Pool> = Box<
    dyn BestTransactions<Item = Arc<ValidPoolTransaction<<Pool as TransactionPool>::Transaction>>>,
>;
#[inline]
pub fn lzero_custom_payload<EvmConfig, Pool, Client, F>(
    evm_config: EvmConfig,
    builder_config: EthereumBuilderConfig,
    args: BuildArguments<Pool, Client, EthPayloadBuilderAttributes, EthBuiltPayload>,
    initialized_cfg: CfgEnvWithHandlerCfg,
    initialized_block_env: BlockEnv,
    best_txs: F,
) -> Result<BuildOutcome<EthBuiltPayload>, PayloadBuilderError>
where
    EvmConfig: ConfigureEvm<Header = Header, Transaction = TransactionSigned>,
    Client: StateProviderFactory + ChainSpecProvider<ChainSpec = ChainSpec>,
    Pool: TransactionPool<Transaction: PoolTransaction<Consensus = TransactionSigned>>,
    F: FnOnce(BestTransactionsAttributes) -> BestTransactionsIter<Pool>,
{
    let BuildArguments { client, pool, mut cached_reads, config, cancel, best_payload } = args;

    let chain_spec = client.chain_spec();
    let state_provider = client.state_by_block_hash(config.parent_header.as_ref().hash())?;
    let state = StateProviderDatabase::new(state_provider);
    let mut db =
        State::builder().with_database(cached_reads.as_db_mut(state)).with_bundle_update().build();
    let PayloadConfig { parent_header, attributes } = config;

    debug!(target: "payload_builder", id=%attributes.id, parent_header = ?parent_header.as_ref().hash(), parent_number = parent_header.number, "building new payload");
    let mut cumulative_gas_used = 0;
    let mut sum_blob_gas_used = 0;
    let block_gas_limit: u64 = initialized_block_env.gas_limit.to::<u64>();
    let base_fee = initialized_block_env.basefee.to::<u64>();

    let mut executed_txs = Vec::new();
    let mut executed_senders = Vec::new();

    let mut best_txs = best_txs(BestTransactionsAttributes::new(
        base_fee,
        initialized_block_env.get_blob_gasprice().map(|gasprice| gasprice as u64),
    ));
    let mut total_fees = U256::ZERO;

    let block_number = initialized_block_env.number.to::<u64>();

    let mut system_caller = SystemCaller::new(evm_config.clone(), chain_spec.clone());

    // apply eip-4788 pre block contract call
    system_caller
        .pre_block_beacon_root_contract_call(
            &mut db,
            &initialized_cfg,
            &initialized_block_env,
            attributes.parent_beacon_block_root,
        )
        .map_err(|err| {
            warn!(target: "payload_builder",
                parent_hash=%parent_header.as_ref().hash(),
                %err,
                "failed to apply beacon root contract call for payload"
            );
            PayloadBuilderError::Internal(err.into())
        })?;

    // apply eip-2935 blockhashes update
    system_caller.pre_block_blockhashes_contract_call(
        &mut db,
        &initialized_cfg,
        &initialized_block_env,
        parent_header.as_ref().hash(),
    )
        .map_err(|err| {
            warn!(target: "payload_builder", parent_hash=%parent_header.as_ref().hash(), %err, "failed to update parent header blockhashes for payload");
            PayloadBuilderError::Internal(err.into())
        })?;

    let env = EnvWithHandlerCfg::new_with_cfg_env(
        initialized_cfg.clone(),
        initialized_block_env.clone(),
        TxEnv::default(),
    );
    let mut evm = evm_config.evm_with_env(&mut db, env);

    let mut receipts = Vec::new();
    while let Some(pool_tx) = best_txs.next() {
        // ensure we still have capacity for this transaction
        if cumulative_gas_used + pool_tx.gas_limit() > block_gas_limit {
            // we can't fit this transaction into the block, so we need to mark it as invalid
            // which also removes all dependent transaction from the iterator before we can
            // continue
            best_txs.mark_invalid(
                &pool_tx,
                InvalidPoolTransactionError::ExceedsGasLimit(pool_tx.gas_limit(), block_gas_limit),
            );
            continue
        }

        // check if the job was cancelled, if so we can exit early
        if cancel.is_cancelled() {
            return Ok(BuildOutcome::Cancelled)
        }

        // convert tx to a signed transaction
        let tx = pool_tx.to_consensus();

        // There's only limited amount of blob space available per block, so we need to check if
        // the EIP-4844 can still fit in the block
        if let Some(blob_tx) = tx.transaction.as_eip4844() {
            let tx_blob_gas = blob_tx.blob_gas();
            if sum_blob_gas_used + tx_blob_gas > MAX_DATA_GAS_PER_BLOCK {
                // we can't fit this _blob_ transaction into the block, so we mark it as
                // invalid, which removes its dependent transactions from
                // the iterator. This is similar to the gas limit condition
                // for regular transactions above.
                trace!(target: "payload_builder", tx=?tx.hash, ?sum_blob_gas_used, ?tx_blob_gas, "skipping blob transaction because it would exceed the max data gas per block");
                best_txs.mark_invalid(
                    &pool_tx,
                    InvalidPoolTransactionError::ExceedsGasLimit(
                        tx_blob_gas,
                        MAX_DATA_GAS_PER_BLOCK,
                    ),
                );
                continue
            }
        }

        // Configure the environment for the tx.
        *evm.tx_mut() = evm_config.tx_env(tx.as_signed(), tx.signer());

        let ResultAndState { result, state } = match evm.transact() {
            Ok(res) => res,
            Err(err) => {
                match err {
                    EVMError::Transaction(err) => {
                        if matches!(err, InvalidTransaction::NonceTooLow { .. }) {
                            // if the nonce is too low, we can skip this transaction
                            trace!(target: "payload_builder", %err, ?tx, "skipping nonce too low transaction");
                        } else {
                            // if the transaction is invalid, we can skip it and all of its
                            // descendants
                            trace!(target: "payload_builder", %err, ?tx, "skipping invalid transaction and its descendants");
                            best_txs.mark_invalid(
                                &pool_tx,
                                InvalidPoolTransactionError::Consensus(
                                    InvalidTransactionError::TxTypeNotSupported,
                                ),
                            );
                        }
                        continue
                    }
                    err => {
                        // this is an error that we should treat as fatal for this attempt
                        return Err(PayloadBuilderError::EvmExecutionError(err))
                    }
                }
            }
        };

        // commit changes
        evm.db_mut().commit(state);

        // add to the total blob gas used if the transaction successfully executed
        if let Some(blob_tx) = tx.transaction.as_eip4844() {
            let tx_blob_gas = blob_tx.blob_gas();
            sum_blob_gas_used += tx_blob_gas;

            // if we've reached the max data gas per block, we can skip blob txs entirely
            if sum_blob_gas_used == MAX_DATA_GAS_PER_BLOCK {
                best_txs.skip_blobs();
            }
        }

        let gas_used = result.gas_used();

        // add gas used by the transaction to cumulative gas used, before creating the receipt
        cumulative_gas_used += gas_used;

        // Push transaction changeset and calculate header bloom filter for receipt.
        #[allow(clippy::needless_update)] // side-effect of optimism fields
        receipts.push(Some(Receipt {
            tx_type: tx.tx_type(),
            success: result.is_success(),
            cumulative_gas_used,
            logs: result.into_logs().into_iter().collect(),
            ..Default::default()
        }));

        // update add to total fees
        let miner_fee = tx
            .effective_tip_per_gas(base_fee)
            .expect("fee is always valid; execution succeeded");
        total_fees += U256::from(miner_fee) * U256::from(gas_used);

        // append sender and transaction to the respective lists
        executed_senders.push(tx.signer());
        executed_txs.push(tx.into_signed());
    }

    // check if we have a better block
    if !is_better_payload(best_payload.as_ref(), total_fees) {
        // Release db
        drop(evm);

        // can skip building the block
        return Ok(BuildOutcome::Aborted { fees: total_fees, cached_reads })
    }

    // calculate the requests and the requests root
    let requests = if chain_spec.is_prague_active_at_timestamp(attributes.timestamp) {
        let deposit_requests = parse_deposits_from_receipts(&chain_spec, receipts.iter().flatten())
            .map_err(|err| PayloadBuilderError::Internal(RethError::Execution(err.into())))?;

        let mut requests = Requests::default();

        if !deposit_requests.is_empty() {
            requests.push_request_with_type(eip6110::DEPOSIT_REQUEST_TYPE, deposit_requests);
        }

        requests.extend(
            system_caller
                .apply_post_execution_changes(&mut evm)
                .map_err(|err| PayloadBuilderError::Internal(err.into()))?,
        );

        Some(requests)
    } else {
        None
    };

    // Release db
    drop(evm);

    let withdrawals_root =
        commit_withdrawals(&mut db, &chain_spec, attributes.timestamp, &attributes.withdrawals)?;

    // Add the contracts to the state
    if let Some(transition_state) = db.transition_state.clone() {
        for (address, transition_account) in &transition_state.transitions {
            if let Some(info) = &transition_account.info {
                if let Some(code) = &info.code {
                    let code_hash = info.code_hash;
                    if !db.bundle_state.contracts.contains_key(&code_hash) {
                        db.bundle_state.contracts.insert(code_hash, code.clone());
                        let bundle_account = BundleAccount {
                            info: Some(info.clone()),
                            original_info: transition_account.previous_info.clone(),
                            storage: transition_account.storage.clone(),
                            status: transition_account.status.clone(),
                        };

                        db.bundle_state.state.insert(address.clone(), bundle_account);
                    }
                }
            }
        }
    }

    // merge all transitions into bundle state, this would apply the withdrawal balance changes
    // and 4788 contract call
    db.merge_transitions(BundleRetention::Reverts);

    let requests_hash = requests.as_ref().map(|requests| requests.requests_hash());
    let execution_outcome = ExecutionOutcome::new(
        db.take_bundle(),
        vec![receipts].into(),
        block_number,
        vec![requests.clone().unwrap_or_default()],
    );
    let receipts_root =
        execution_outcome.ethereum_receipts_root(block_number).expect("Number is in range");
    let logs_bloom = execution_outcome.block_logs_bloom(block_number).expect("Number is in range");

    // calculate the state root
    let hashed_state = db.database.db.hashed_post_state(execution_outcome.state());
    let (state_root, trie_output) = {
        db.database.inner().state_root_with_updates(hashed_state.clone()).inspect_err(|err| {
            warn!(target: "payload_builder",
                parent_hash=%parent_header.as_ref().hash(),
                %err,
                "failed to calculate state root for payload"
            );
        })?
    };

    // create the block header
    let transactions_root = proofs::calculate_transaction_root(&executed_txs);

    // initialize empty blob sidecars at first. If cancun is active then this will
    let mut blob_sidecars = Vec::new();
    let mut excess_blob_gas = None;
    let mut blob_gas_used = None;

    // only determine cancun fields when active
    if chain_spec.is_cancun_active_at_timestamp(attributes.timestamp) {
        // grab the blob sidecars from the executed txs
        blob_sidecars = pool
            .get_all_blobs_exact(
                executed_txs.iter().filter(|tx| tx.is_eip4844()).map(|tx| tx.hash()).collect(),
            )
            .map_err(PayloadBuilderError::other)?;

        excess_blob_gas = if chain_spec.is_cancun_active_at_timestamp(parent_header.timestamp) {
            let blob_params = if chain_spec.is_prague_active_at_timestamp(parent_header.timestamp) {
                BlobParams::prague()
            } else {
                // cancun
                BlobParams::cancun()
            };
            parent_header.next_block_excess_blob_gas(blob_params)
        } else {
            // for the first post-fork block, both parent.blob_gas_used and
            // parent.excess_blob_gas are evaluated as 0
            Some(alloy_eips::eip4844::calc_excess_blob_gas(0, 0))
        };

        blob_gas_used = Some(sum_blob_gas_used);
    }

    let header = Header {
        parent_hash: parent_header.as_ref().hash(),
        ommers_hash: EMPTY_OMMER_ROOT_HASH,
        beneficiary: initialized_block_env.coinbase,
        state_root,
        transactions_root,
        receipts_root,
        withdrawals_root,
        logs_bloom,
        timestamp: attributes.timestamp,
        mix_hash: attributes.prev_randao,
        nonce: BEACON_NONCE.into(),
        base_fee_per_gas: Some(base_fee),
        number: parent_header.number + 1,
        gas_limit: block_gas_limit,
        difficulty: U256::ZERO,
        gas_used: cumulative_gas_used,
        extra_data: builder_config.extra_data,
        parent_beacon_block_root: attributes.parent_beacon_block_root,
        blob_gas_used,
        excess_blob_gas,
        requests_hash,
    };

    let withdrawals = chain_spec
        .is_shanghai_active_at_timestamp(attributes.timestamp)
        .then(|| attributes.withdrawals.clone());

    // seal the block
    let block = Block {
        header,
        body: BlockBody { transactions: executed_txs, ommers: vec![], withdrawals },
    };

    let sealed_block = Arc::new(block.seal_slow());
    debug!(target: "payload_builder", id=%attributes.id, sealed_block_header = ?sealed_block.header, "sealed built block");

    // create the executed block data
    let executed = ExecutedBlock {
        block: sealed_block.clone(),
        senders: Arc::new(executed_senders),
        execution_output: Arc::new(execution_outcome),
        hashed_state: Arc::new(hashed_state),
        trie: Arc::new(trie_output),
    };

    let mut payload =
        EthBuiltPayload::new(attributes.id, sealed_block, total_fees, Some(executed), requests);

    // extend the payload with the blob sidecars from the executed txs
    payload.extend_sidecars(blob_sidecars.into_iter().map(Arc::unwrap_or_clone));

    Ok(BuildOutcome::Better { payload, cached_reads })
}