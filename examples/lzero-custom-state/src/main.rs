#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use std::{path::Path, sync::Arc};

use alloy_consensus::{BlockHeader as Header};
use alloy_genesis::Genesis;
use alloy_primitives::{address, Address, Bytes, U256, B256};
use alloy_rpc_types_eth::{Filter, FilteredParams};
use jsonrpsee::core::RpcResult;
use jsonrpsee::proc_macros::rpc;
use reth::{
    api::NodeTypes,
    builder::{
        components::{ExecutorBuilder, PayloadServiceBuilder, PoolBuilder},
        BuilderContext, NodeBuilder, FullNodeTypes
    },
    chainspec::{ChainSpec, Chain},
    cli::Cli,
    primitives::EthPrimitives,
    providers::CanonStateSubscriptions,
    payload::{EthBuiltPayload, EthPayloadBuilderAttributes},
    revm::{
        handler::register::EvmHandler,
        inspector_handle_register,
        precompile::{Precompile, PrecompileOutput, PrecompileSpecId},
        primitives::{BlockEnv, CfgEnvWithHandlerCfg, Env, PrecompileResult, TxEnv, EVMError},
        Database, Evm, EvmBuilder, GetInspector,
    },
    rpc::types::engine::PayloadAttributes,
    tasks::TaskManager,
    transaction_pool::{
        blobstore::InMemoryBlobStore, EthTransactionPool, TransactionValidationTaskExecutor,
        PoolTransaction,
    },
};
use reth::rpc::eth::RpcNodeCore;
use reth_chainspec::ChainSpecBuilder;
use reth_db::{open_db_read_only, DatabaseEnv};
use reth_node_api::{ConfigureEvm, ConfigureEvmEnv, NextBlockEnvAttributes, PayloadTypes};
use reth_node_core::{args::RpcServerArgs, node_config::NodeConfig};
use reth_node_ethereum::{
    node::{EthereumAddOns, EthereumNode},
    BasicBlockExecutorProvider, EthExecutionStrategyFactory, EthereumPayloadBuilder,
};
use reth_node_types::FullNode;
use reth_primitives::{BlockExt, SealedHeader, TransactionSigned};
use reth_provider::{
    providers::StaticFileProvider, AccountReader, BlockReader, BlockSource, HeaderProvider,
    ProviderFactory, ReceiptProvider, StateProvider, TransactionsProvider,
};
use reth_transaction_pool::{
    maintain::{backup_local_transactions_task, maintain_transaction_pool_future, LocalTransactionBackupConfig},
    Pool, PoolConfig
};
use reth_tracing::{RethTracer, Tracer};
use reth_keystore::LocalKeystore;

// ======================================
// Custom EVM Configuration (MyEvmConfig)
// ======================================

#[derive(Debug, Clone)]
pub struct MyEvmConfig {
    inner: reth_evm_ethereum::EthEvmConfig,
}

impl MyEvmConfig {
    pub fn new(chain_spec: Arc<ChainSpec>) -> Self {
        Self { inner: reth_evm_ethereum::EthEvmConfig::new(chain_spec) }
    }

    // This is our custom precompile logic that can set arbitrary storage.
    // Data format: [0..32 bytes: contract_address padded][32 bytes: slot_key][32 bytes: value]
    fn my_precompile(data: &Bytes, gas: u64, env: &Env) -> PrecompileResult {
        if data.len() < 96 {
            return Err(EVMError::OutOfGas);
        }

        // Extract the target contract address from the last 20 bytes of the first 32-byte word
        let mut addr_buf = [0u8; 20];
        addr_buf.copy_from_slice(&data[12..32]);
        let contract_addr = Address::from(addr_buf);

        let slot_key = B256::from_slice(&data[32..64]);
        let value = B256::from_slice(&data[64..96]);

        // Gas cost (simplified)
        let gas_cost = 20_000;
        if gas < gas_cost {
            return Err(EVMError::OutOfGas);
        }

        // Obtain a handler that can modify state.
        // This assumes `env.ctx` is set up to provide a mutable handler.
        // In reality, you must ensure your EVM builder stores a handler reference in `env.ctx`.
        let handler = env.ctx
            .as_ref()
            .ok_or(EVMError::FatalExternalError)?
            .handler_mut();

        // Assume we have a method: handler.set_storage(address, slot_key, value)
        // You must implement this in your custom EVM handler.
        handler.set_storage(contract_addr, slot_key, value);

        Ok(PrecompileOutput::new(gas_cost, Bytes::new()))
    }

    fn set_precompiles<EXT, DB>(handler: &mut EvmHandler<EXT, DB>)
    where
        DB: reth::revm::db::Database,
    {
        let spec_id = handler.cfg.spec_id;
        handler.pre_execution.load_precompiles = Arc::new(move || {
            let mut precompiles = reth::revm::primitives::ContextPrecompiles::new(
                PrecompileSpecId::from_spec_id(spec_id),
            );
            // Place our custom precompile at an address, e.g. 0x...0099
            let mut custom_addr = [0u8; 20];
            custom_addr[19] = 0x99; // last byte set to 0x99
            let precompile_addr = Address::from(custom_addr);

            precompiles.extend([(
                precompile_addr,
                Precompile::Env(MyEvmConfig::my_precompile).into(),
            )]);
            precompiles
        });
    }
}

impl ConfigureEvmEnv for MyEvmConfig {
    type Header = Header;
    type Transaction = TransactionSigned;
    type Error = std::convert::Infallible;

    fn fill_tx_env(&self, tx_env: &mut TxEnv, transaction: &TransactionSigned, sender: Address) {
        self.inner.fill_tx_env(tx_env, transaction, sender);
    }

    fn fill_tx_env_system_contract_call(&self, env: &mut Env, caller: Address, contract: Address, data: Bytes) {
        self.inner.fill_tx_env_system_contract_call(env, caller, contract, data);
    }

    fn fill_cfg_env(&self, cfg_env: &mut CfgEnvWithHandlerCfg, header: &Self::Header, total_difficulty: U256) {
        self.inner.fill_cfg_env(cfg_env, header, total_difficulty);
    }

    fn next_cfg_and_block_env(&self, parent: &Self::Header, attributes: NextBlockEnvAttributes) -> Result<(CfgEnvWithHandlerCfg, BlockEnv), Self::Error> {
        self.inner.next_cfg_and_block_env(parent, attributes)
    }
}

impl ConfigureEvm for MyEvmConfig {
    type DefaultExternalContext<'a> = ();

    fn evm<DB: Database>(&self, db: DB) -> Evm<'_, Self::DefaultExternalContext<'_>, DB> {
        EvmBuilder::default()
            .with_db(db)
            .append_handler_register(MyEvmConfig::set_precompiles)
            .build()
    }

    fn evm_with_inspector<DB, I>(&self, db: DB, inspector: I) -> Evm<'_, I, DB>
    where
        DB: Database,
        I: GetInspector<DB>,
    {
        EvmBuilder::default()
            .with_db(db)
            .with_external_context(inspector)
            .append_handler_register(MyEvmConfig::set_precompiles)
            .append_handler_register(inspector_handle_register)
            .build()
    }

    fn default_external_context<'a>(&self) -> Self::DefaultExternalContext<'a> {}
}

// ==========================
// Custom Executor & Payload
// ==========================

#[derive(Debug, Default, Clone, Copy)]
pub struct MyExecutorBuilder;

impl<Node> ExecutorBuilder<Node> for MyExecutorBuilder
where
    Node: FullNodeTypes<Types: NodeTypes<ChainSpec = ChainSpec, Primitives = EthPrimitives>>,
{
    type EVM = MyEvmConfig;
    type Executor = BasicBlockExecutorProvider<EthExecutionStrategyFactory<Self::EVM>>;

    async fn build_evm(self, ctx: &BuilderContext<Node>) -> eyre::Result<(Self::EVM, Self::Executor)> {
        Ok((
            MyEvmConfig::new(ctx.chain_spec()),
            BasicBlockExecutorProvider::new(EthExecutionStrategyFactory::new(
                ctx.chain_spec(),
                MyEvmConfig::new(ctx.chain_spec()),
            )),
        ))
    }
}

#[derive(Debug, Default, Clone)]
pub struct MyPayloadBuilder {
    inner: EthereumPayloadBuilder,
}

impl<Types, Node, Pool> PayloadServiceBuilder<Node, Pool> for MyPayloadBuilder
where
    Types: reth_node_api::NodeTypesWithEngine<ChainSpec = ChainSpec, Primitives = EthPrimitives>,
    Node: FullNodeTypes<Types = Types>,
    Pool: EthTransactionPool<Transaction = TransactionSigned> + Unpin + 'static,
    Types::Engine: PayloadTypes<
        BuiltPayload = EthBuiltPayload,
        PayloadAttributes = PayloadAttributes,
        PayloadBuilderAttributes = EthPayloadBuilderAttributes,
    >,
{
    async fn spawn_payload_service(
        self,
        ctx: &BuilderContext<Node>,
        pool: Pool,
    ) -> eyre::Result<reth::payload::PayloadBuilderHandle<Types::Engine>> {
        self.inner.spawn(MyEvmConfig::new(ctx.chain_spec()), ctx, pool)
    }
}

// ===========================
// Custom RPC for State Update
// ===========================

#[rpc(server)]
pub trait DebugState {
    #[method(name = "debug_setStorageSlot")]
    async fn debug_set_storage_slot(&self, contract: Address, slot: B256, value: B256) -> RpcResult<String>;
}

pub struct DebugStateRpc<P> {
    pool: Arc<P>,
    keystore: Arc<LocalKeystore>,
    precompile_addr: Address,
    from_address: Address, // an address from which we send these transactions
    chain_id: u64,
}

impl<P> DebugStateRpc<P> {
    pub fn new(pool: Arc<P>, keystore: Arc<LocalKeystore>, from: Address, chain_id: u64) -> Self {
        let mut addr_bytes = [0u8; 20];
        addr_bytes[19] = 0x99;
        let precompile_addr = Address::from(addr_bytes);

        Self {
            pool,
            keystore,
            precompile_addr,
            from_address: from,
            chain_id,
        }
    }
}

#[async_trait::async_trait]
impl<P> DebugStateServer for DebugStateRpc<P>
where
    P: EthTransactionPool<Transaction = TransactionSigned> + 'static + Send + Sync,
{
    async fn debug_set_storage_slot(&self, contract: Address, slot: B256, value: B256) -> RpcResult<String> {
        // Encode calldata: 32 bytes contract (padded), 32 bytes slot, 32 bytes value
        let mut data = Vec::with_capacity(96);
        let mut padded_contract = [0u8; 32];
        padded_contract[12..32].copy_from_slice(contract.as_bytes());
        data.extend_from_slice(&padded_contract);
        data.extend_from_slice(slot.as_bytes());
        data.extend_from_slice(value.as_bytes());

        // For simplicity, assume we fetch nonce = 0, gas_price = 1gwei, etc.
        // In reality, you'd query the provider for nonce and gas price.
        let nonce = 0;
        let gas_price: U256 = 1_000_000_000u64.into();
        let gas_limit = 5_000_000u64;

        // Construct a transaction calling the precompile
        // A helper constructor might be needed. For demonstration:
        let tx = TransactionSigned::new_call(
            self.from_address,
            self.precompile_addr,
            U256::zero(),
            gas_limit,
            gas_price,
            data.into(),
            self.chain_id,
            &*self.keystore, // signing with keystore
            nonce,
        ).map_err(|e| jsonrpsee::core::Error::Custom(e.to_string()))?;

        // Submit to pool
        self.pool.add_transaction(tx.clone())
            .await
            .map_err(|e| jsonrpsee::core::Error::Custom(e.to_string()))?;

        Ok(format!("Transaction submitted: {:?}", tx.hash()))
    }
}

// ===================================
// Main Node Setup with Custom Config
// ===================================

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let _guard = reth_tracing::RethTracer::new().init()?;
    let tasks = TaskManager::current();

    // create a custom chain spec
    let spec = ChainSpecBuilder::mainnet()
        .chain(Chain::mainnet())
        .genesis(Genesis::default())
        .london_activated()
        .paris_activated()
        .shanghai_activated()
        .cancun_activated()
        .build();

    let node_config =
        NodeConfig::test().with_rpc(RpcServerArgs::default().with_http()).with_chain(spec);

    // We need a keystore for signing transactions for our RPC calls
    // In a real setup, initialize the keystore and load some keys.
    let keystore = Arc::new(LocalKeystore::in_memory());
    // Insert a key and store the from_address for demonstration
    let from_address = Address::random();
    keystore.import_plain_key([0x11u8;32], from_address)?;

    let builder = NodeBuilder::new(node_config)
        .testing_node(tasks.executor())
        // use default ethereum node types
        .with_types::<EthereumNode>()
        // configure components: custom executor and payload builder
        .with_components(
            EthereumNode::components()
                .executor(MyExecutorBuilder::default())
                .payload(MyPayloadBuilder::default()),
        )
        .with_add_ons(EthereumAddOns::default());

    // Launch the node
    let handle = builder.launch().await.unwrap();

    // Register our custom RPC method
    {
        let pool = handle.pool().clone();
        let chain_id = 1u64; // Mainnet chain_id for demonstration
        let debug_state_rpc = DebugStateRpc::new(pool, keystore.clone(), from_address, chain_id);
        let mut rpc_module = jsonrpsee::RpcModule::new(());
        rpc_module.merge(debug_state_rpc.into_rpc()).unwrap();

        // Integrate rpc_module into the running node's RPC server
        // For demonstration, this might require access to node's RPC server builder
        // handle.rpc_server().register_module(rpc_module);
        // (Adjust this depending on `reth` version and how you add custom modules.)
    }

    println!("Node started");

    handle.node_exit_future.await;

    Ok(())
}