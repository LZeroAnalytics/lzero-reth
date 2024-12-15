#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use alloy_consensus::{Header, Transaction};
use alloy_genesis::{ChainConfig, Genesis, GenesisAccount};
use alloy_primitives::{hex, Address, Bytes, U256, B256};
use reth::{builder::{
    components::{ExecutorBuilder, PayloadServiceBuilder},
    BuilderContext, NodeBuilder,
}, payload::{EthBuiltPayload, EthPayloadBuilderAttributes}, revm, tasks::TaskManager, transaction_pool::{PoolTransaction, TransactionPool}};
use reth_chainspec::{Chain, ChainSpec};
use reth_evm_ethereum::EthEvmConfig;
use reth_node_api::{
    ConfigureEvm, ConfigureEvmEnv, FullNodeTypes, NextBlockEnvAttributes, NodeTypes,
    NodeTypesWithEngine, PayloadTypes,
};
use reth_node_core::{args::RpcServerArgs, node_config::NodeConfig};
use reth_node_ethereum::{
    node::{EthereumAddOns, EthereumPayloadBuilder, EthereumNode},
    EthExecutionStrategyFactory,
};
use reth_primitives::{EthPrimitives, TransactionSigned};
use reth_tracing::{RethTracer, Tracer};
use std::{convert::Infallible, sync::Arc};

// Imports from your execute.rs logic
use reth_evm::execute::{
    BlockExecutorProvider, BlockExecutionError, BlockExecutionInput, BlockExecutionOutput,
    BlockExecutionStrategy, ExecuteOutput, ExecutionOutcome,
};
use reth_evm::execute::{BasicBlockExecutorProvider, BasicBlockExecutor, BasicBatchExecutor};
use reth_primitives_traits::NodePrimitives;
use reth_primitives::{BlockWithSenders, Receipt};
use revm_primitives::db::Database;
use reth_storage_errors::provider::ProviderError;
use core::fmt::Display;
use reth_consensus::ConsensusError;

use serde_json::from_str;


// A simple EVM config that mostly delegates to EthEvmConfig.
#[derive(Debug, Clone)]
pub struct MyEvmConfig {
    inner: EthEvmConfig,
}

impl MyEvmConfig {
    pub const fn new(chain_spec: Arc<ChainSpec>) -> Self {
        Self { inner: EthEvmConfig::new(chain_spec) }
    }
}

impl ConfigureEvmEnv for MyEvmConfig {
    type Header = Header;
    type Transaction = TransactionSigned;
    type Error = Infallible;

    fn fill_tx_env(&self, tx_env: &mut revm::primitives::TxEnv, transaction: &Self::Transaction, sender: Address) {
        self.inner.fill_tx_env(tx_env, transaction, sender);
    }

    fn fill_tx_env_system_contract_call(
        &self,
        env: &mut revm::primitives::Env,
        caller: Address,
        contract: Address,
        data: Bytes,
    ) {
        self.inner.fill_tx_env_system_contract_call(env, caller, contract, data);
    }

    fn fill_cfg_env(&self, cfg_env: &mut revm::primitives::CfgEnvWithHandlerCfg, header: &Self::Header, total_difficulty: U256) {
        self.inner.fill_cfg_env(cfg_env, header, total_difficulty);
    }

    fn next_cfg_and_block_env(
        &self,
        parent: &Self::Header,
        attributes: NextBlockEnvAttributes,
    ) -> Result<(revm::primitives::CfgEnvWithHandlerCfg, revm::primitives::BlockEnv), Self::Error> {
        self.inner.next_cfg_and_block_env(parent, attributes)
    }
}

impl ConfigureEvm for MyEvmConfig {
    type DefaultExternalContext<'a> = ();

    fn evm<DB: Database>(&self, db: DB) -> revm::Evm<'_, Self::DefaultExternalContext<'_>, DB> {
        revm::EvmBuilder::default().with_db(db).build()
    }

    fn evm_with_inspector<DB, I>(&self, db: DB, inspector: I) -> revm::Evm<'_, I, DB>
    where
        DB: Database,
        I: revm::GetInspector<DB>,
    {
        revm::EvmBuilder::default().with_db(db).with_external_context(inspector).build()
    }

    fn default_external_context<'a>(&self) -> Self::DefaultExternalContext<'a> {}
}

// We need NodePrimitives to define block/receipt types.
// We'll rely on `EthereumNode` which typically uses `EthPrimitives`.
type Primitives = <EthereumNode as NodeTypes>::Primitives; // EthPrimitives

/// A simple strategy that just prints out any custom transactions found in the block.
/// No state modifications, no receipts, just logging.
#[derive(Clone)]
pub struct SimpleStrategy<DB> {
    db: DB,
}

impl<DB> BlockExecutionStrategy for SimpleStrategy<DB>
where
    DB: Database<Error: Into<ProviderError> + Display>,
{
    type DB = DB;
    type Primitives = Primitives;
    type Error = BlockExecutionError;

    fn apply_pre_execution_changes(
        &mut self,
        _block: &BlockWithSenders<<Self::Primitives as NodePrimitives>::Block>,
        _total_difficulty: U256,
    ) -> Result<(), Self::Error> {
        // No changes needed
        Ok(())
    }

    fn execute_transactions(
        &mut self,
        block: &BlockWithSenders<<Self::Primitives as NodePrimitives>::Block>,
        _total_difficulty: U256,
    ) -> Result<ExecuteOutput<<Self::Primitives as NodePrimitives>::Receipt>, Self::Error> {
        for tx in block.body.transactions.iter() {
            if tx.transaction.to() == Some("0x000000000000000000000000000000000000d00d".parse().unwrap()) {
                let data = tx.input();
                println!("Custom transaction detected!");
                if data.len() < 20 {
                    println!("Invalid data length for custom tx");
                    continue;
                }
                let target_addr = Address::from_slice(&data[0..20]);
                println!("Target contract: {:?}", target_addr);

                let mut offset = 20;
                while offset + 64 <= data.len() {
                    let slot = &data[offset..offset+32];
                    let value = &data[offset+32..offset+64];
                    offset += 64;

                    println!("  Slot: 0x{}, Value: 0x{}",
                          hex::encode(slot),
                          hex::encode(value));
                }
            }
        }

        // Return dummy receipts and gas_used
        Ok(ExecuteOutput { receipts: vec![], gas_used: 0 })
    }

    fn apply_post_execution_changes(
        &mut self,
        _block: &BlockWithSenders<<Self::Primitives as NodePrimitives>::Block>,
        _total_difficulty: U256,
        _receipts: &[<Self::Primitives as NodePrimitives>::Receipt],
    ) -> Result<alloy_eips::eip7685::Requests, Self::Error> {
        Ok(Default::default())
    }

    fn state_ref(&self) -> &revm::State<DB> {
        // Just return a default state reference (no changes)
        // In reality you'd store some state. For a basic example, we can return an empty state.
        panic!("No state implemented for this simple example")
    }

    fn state_mut(&mut self) -> &mut revm::State<DB> {
        panic!("No state implemented for this simple example")
    }

    fn finish(&mut self) -> revm::db::states::bundle_state::BundleState {
        // Return an empty bundle state
        Default::default()
    }

    fn validate_block_post_execution(
        &self,
        _block: &BlockWithSenders<<Self::Primitives as NodePrimitives>::Block>,
        _receipts: &[<Self::Primitives as NodePrimitives>::Receipt],
        _requests: &alloy_eips::eip7685::Requests,
    ) -> Result<(), ConsensusError> {
        Ok(())
    }
}

// A factory that creates SimpleStrategy instances
#[derive(Clone)]
pub struct SimpleStrategyFactory;

impl reth_evm::execute::BlockExecutionStrategyFactory for SimpleStrategyFactory {
    type Primitives = Primitives;

    type Strategy<DB: Database<Error: Into<ProviderError> + Display>> = SimpleStrategy<DB>;

    fn create_strategy<DB>(&self, db: DB) -> Self::Strategy<DB>
    where
        DB: Database<Error: Into<ProviderError> + Display>,
    {
        SimpleStrategy { db }
    }
}

// Our ExecutorBuilder that uses SimpleStrategyFactory
#[derive(Debug, Default, Clone, Copy)]
pub struct MyExecutorBuilder;

impl<Node> ExecutorBuilder<Node> for MyExecutorBuilder
where
    Node: FullNodeTypes<Types: NodeTypes<ChainSpec = ChainSpec, Primitives = EthPrimitives>>,
{
    type EVM = MyEvmConfig;
    type Executor = BasicBlockExecutorProvider<SimpleStrategyFactory>;

    async fn build_evm(
        self,
        ctx: &BuilderContext<Node>,
    ) -> eyre::Result<(Self::EVM, Self::Executor)> {
        let evm_config = MyEvmConfig::new(ctx.chain_spec());
        let provider = reth_evm::execute::BasicBlockExecutorProvider::new(SimpleStrategyFactory);
        Ok((evm_config, provider))
    }
}

/// A basic payload builder that does nothing special
#[derive(Debug, Default, Clone)]
pub struct MyPayloadBuilder {
    inner: EthereumPayloadBuilder,
}

use reth::payload::PayloadBuilderHandle;
impl<Types, Node, Pool> PayloadServiceBuilder<Node, Pool> for MyPayloadBuilder
where
    Types: NodeTypesWithEngine<ChainSpec = ChainSpec, Primitives = EthPrimitives>,
    Node: FullNodeTypes<Types = Types>,
    Pool: TransactionPool<Transaction: PoolTransaction<Consensus = TransactionSigned>> + Unpin + 'static,
    Types::Engine: PayloadTypes<
        BuiltPayload = EthBuiltPayload,
        PayloadAttributes = reth::rpc::types::engine::PayloadAttributes,
        PayloadBuilderAttributes = EthPayloadBuilderAttributes,
    >,
{
    async fn spawn_payload_service(
        self,
        ctx: &BuilderContext<Node>,
        pool: Pool,
    ) -> eyre::Result<PayloadBuilderHandle<Types::Engine>> {
        self.inner.spawn(MyEvmConfig::new(ctx.chain_spec()), ctx, pool)
    }
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let _guard = RethTracer::new().init()?;

    let tasks = TaskManager::current();

    let genesis_json = r#"
    {
        "nonce": "0x0000000000000042",
        "difficulty": "34747478",
        "mixHash": "0x123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234",
        "coinbase": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "timestamp": "0x123456",
        "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
        "extraData": "0xfafbfcfd",
        "gasLimit": "0x2fefd8",
        "alloc": {
            "0x8943545177806ED17B9F23F0a21ee5948eCaa776": {
              "balance": "1000000000000000000000"
            }
        },
        "config": {
            "ethash": {},
            "chainId": 10,
            "homesteadBlock": 0,
            "eip150Block": 0,
            "eip155Block": 0,
            "eip158Block": 0,
            "byzantiumBlock": 0,
            "constantinopleBlock": 0,
            "petersburgBlock": 0,
            "istanbulBlock": 0
        }
    }
    "#;

    let genesis: Genesis = from_str(genesis_json)?;

    let spec = ChainSpec::builder()
        .chain(Chain::dev())
        .genesis(genesis)
        .london_activated()
        .paris_activated()
        .shanghai_activated()
        .cancun_activated()
        .build();

    let node_config =
        NodeConfig::test().with_rpc(RpcServerArgs::default().with_http()).with_chain(spec);

    let handle = NodeBuilder::new(node_config)
        .testing_node(tasks.executor())
        .with_types::<EthereumNode>()
        .with_components(
            EthereumNode::components()
                .executor(MyExecutorBuilder::default())
                .payload(MyPayloadBuilder::default()),
        )
        .with_add_ons(EthereumAddOns::default())
        .launch()
        .await
        .unwrap();

    println!("Node started. When a custom transaction with the sentinel address is included in a block, its data will be printed.");

    handle.node_exit_future.await
}
