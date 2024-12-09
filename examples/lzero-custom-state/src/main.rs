use reth::{
    builder::components::ExecutorBuilder,
    builder::BuilderContext,
    primitives::{EthPrimitives},
    revm::Database,
};
use reth_node_api::{FullNodeTypes, NodeTypes};
use reth_transaction_pool::PoolTransaction;
use eyre::Result;
use std::sync::Arc;
use alloy_consensus::Transaction;
use alloy_primitives::{Address, B256, U256};
use reth_chainspec::ChainSpec;

#[derive(Debug, Default, Clone, Copy)]
pub struct CustomExecutorBuilder;

impl<Node> ExecutorBuilder<Node> for CustomExecutorBuilder
where
    Node: FullNodeTypes<Types: NodeTypes<ChainSpec = ChainSpec, Primitives = EthPrimitives>>,
{
    type EVM = MyEvmConfig; // We'll define a trivial EvmConfig
    type Executor = CustomBlockExecutorProvider;

    async fn build_evm(
        self,
        ctx: &BuilderContext<Node>,
    ) -> eyre::Result<(Self::EVM, Self::Executor)> {
        Ok((
            MyEvmConfig::new(ctx.chain_spec()),
            CustomBlockExecutorProvider::new(ctx.chain_spec().clone())
        ))
    }
}

// A trivial EVM config that does nothing special EVM-wise,
// since our custom transactions won't use normal EVM execution.
#[derive(Debug, Clone)]
pub struct MyEvmConfig {
    chain_spec: Arc<ChainSpec>,
}

impl MyEvmConfig {
    pub fn new(chain_spec: Arc<ChainSpec>) -> Self {
        MyEvmConfig { chain_spec }
    }
}

use reth_node_api::{ConfigureEvmEnv, NextBlockEnvAttributes};
use reth::revm::{Evm, EvmBuilder, GetInspector};
use reth::revm::primitives::{TxEnv, Env, BlockEnv, CfgEnvWithHandlerCfg};

impl ConfigureEvmEnv for MyEvmConfig {
    type Header = reth_primitives::SealedHeader; // or appropriate header type
    type Transaction = reth_primitives::TransactionSigned;
    type Error = std::convert::Infallible;

    fn fill_tx_env(&self, _tx_env: &mut TxEnv, _transaction: &Self::Transaction, _sender: Address) {}
    fn fill_tx_env_system_contract_call(&self, _env: &mut Env, _caller: Address, _contract: Address, _data: reth_primitives::Bytecode) {}
    fn fill_cfg_env(&self, _cfg_env: &mut CfgEnvWithHandlerCfg, _header: &Self::Header, _total_difficulty: U256) {}
    fn next_cfg_and_block_env(&self, _parent: &Self::Header, _attributes: NextBlockEnvAttributes) -> Result<(CfgEnvWithHandlerCfg, BlockEnv), Self::Error> {
        // Just return defaults
        Ok((CfgEnvWithHandlerCfg::clone(&CfgEnvWithHandlerCfg { cfg_env: Default::default(), handler_cfg: Default::default() }), BlockEnv::default()))
    }
}

use reth_node_api::ConfigureEvm;
impl ConfigureEvm for MyEvmConfig {
    type DefaultExternalContext<'a> = ();

    fn evm<DB: Database>(&self, db: DB) -> Evm<'_, Self::DefaultExternalContext<'_>, DB> {
        EvmBuilder::default().with_db(db).build()
    }

    fn evm_with_inspector<DB, I>(&self, db: DB, inspector: I) -> Evm<'_, I, DB>
    where
        DB: Database,
        I: GetInspector<DB>,
    {
        EvmBuilder::default().with_db(db).with_external_context(inspector).build()
    }

    fn default_external_context<'a>(&self) -> Self::DefaultExternalContext<'a> {}
}

// Now the custom block executor that checks for our special transaction and applies it.
pub struct CustomBlockExecutorProvider {
    chain_spec: Arc<ChainSpec>,
}

impl CustomBlockExecutorProvider {
    pub fn new(chain_spec: Arc<ChainSpec>) -> Self {
        Self { chain_spec }
    }
}

use reth_provider::{ProviderFactory};

impl CustomBlockExecutorProvider {
    fn execute_custom_transaction<DB: Database>(
        &self,
        db: &mut DB,
        tx: &reth_primitives::TransactionSigned
    ) -> Result<(), String> {
        // Let's define our special sentinel address
        // If 'to' is this address, we treat input data as (target_address (20 bytes), slot (32 bytes), value (32 bytes))
        let sentinel_address = Address::from_slice(&[0u8; 19]); // 0x000...000
        if tx.to() != Some(sentinel_address) {
            // Not our custom tx, skip
            return Err("Not a custom storage-setting transaction".into());
        }

        let data = &tx.input();
        if data.len() < 20 + 32 + 32 {
            return Err("Invalid data length".into());
        }

        let target_addr = Address::from_slice(&data[0..20]);
        let slot = B256::from_slice(&data[20..52]);
        let value = B256::from_slice(&data[52..84]);

        // Now we directly manipulate the state
        // Use the DB and provider to open a write transaction and set storage
        let tx_factory = ProviderFactory::new(db, self.chain_spec.clone(), ());
        let provider_rw = tx_factory.provider_rw().map_err(|e| e.to_string())?;

        // Set the storage slot
        provider_rw.set_storage(&target_addr, slot, value).map_err(|e| e.to_string())?;
        provider_rw.commit().map_err(|e| e.to_string())?;

        Ok(())
    }
}

impl Default for CustomBlockExecutorProvider {
    fn default() -> Self {
        panic!("Use CustomBlockExecutorProvider::new() instead")
    }
}

impl CustomBlockExecutorProvider {
    pub fn execute_block<DB: Database>(
        &self,
        db: &mut DB,
        txs: &[reth_primitives::TransactionSigned]
    ) -> eyre::Result<()> {
        for tx in txs {
            // Try custom execution:
            if let Err(e) = self.execute_custom_transaction(db, tx) {
                // If not custom or failed, fallback to normal EVM execution?
                // For simplicity, let's say we only support custom tx right now.
                // If you want both, integrate with normal EVM calls here.
                if e == "Not a custom storage-setting transaction" {
                    // perform normal EVM execution steps here
                    // For that, you'd instantiate EVM and execute, or delegate to existing logic.
                    // We'll omit that for brevity.
                } else {
                    return Err(eyre::eyre!(e));
                }
            }
        }
        Ok(())
    }
}

use reth_node_ethereum::node::EthereumAddOns;
use reth_node_ethereum::EthereumNode;
use reth_node_core::args::RpcServerArgs;
use reth_node_core::node_config::NodeConfig;
use tokio::runtime::Runtime;
use reth::chainspec::Chain;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let rt = Runtime::new()?;
    rt.block_on(async {
        let chain_spec = ChainSpec::builder()
            .chain(Chain::mainnet())
            .build();

        let node_config = NodeConfig::test()
            .with_rpc(RpcServerArgs::default().with_http())
            .with_chain(chain_spec);

        let handle = reth::builder::NodeBuilder::new(node_config)
            .testing_node(reth::tasks::TaskManager::current().executor())
            .with_types::<EthereumNode>()
            .with_components(
                // Use your custom executor builder here:
                EthereumNode::components()
                    .executor(CustomExecutorBuilder::default())
                    // You can use default payload builder or a custom one:
                    .payload(reth_node_ethereum::node::EthereumPayloadBuilder::default()),
            )
            .with_add_ons(EthereumAddOns::default())
            .launch()
            .await?;

        println!("Node started with custom transaction support");

        handle.node_exit_future.await;
        Ok(())
    })
}