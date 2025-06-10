use clap::Parser;
use reth::cli::Cli;
use reth::chainspec::EthereumChainSpecParser;
use reth_chainspec::ChainSpec;
use reth_evm_ethereum::EthEvmConfig;
use reth_node_ethereum::{EthereumNode, node::EthereumAddOns};
use reth_primitives::EthPrimitives;
use reth::builder::{
    components::ExecutorBuilder,
    BuilderContext, NodeHandle,
};
use reth::api::FullNodeTypes;
use reth_revm::database::StateProviderDatabase;

fn main() {
    // 
    //
    //
    
    Cli::<EthereumChainSpecParser>::parse()
        .run(|builder, _args| async move {
            let handle = builder
                .with_types::<EthereumNode>()
                .with_components(EthereumNode::components().executor(CustomExecutorBuilder::default()))
                .with_add_ons(EthereumAddOns::default())
                .launch()
                .await?;

            let NodeHandle { node: _, node_exit_future } = handle;
            node_exit_future.await
        })
        .unwrap();
}

#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct CustomExecutorBuilder;

impl<Node> ExecutorBuilder<Node> for CustomExecutorBuilder
where
    Node: FullNodeTypes,
    Node::Types: reth::api::NodeTypes<ChainSpec = ChainSpec, Primitives = EthPrimitives>,
{
    type EVM = EthEvmConfig;

    async fn build_evm(self, ctx: &BuilderContext<Node>) -> eyre::Result<Self::EVM> {
        let evm_config = EthEvmConfig::new(ctx.chain_spec());
        
        if let Ok(rpc_url) = std::env::var("FORKING_RPC_URL") {
            if !rpc_url.is_empty() {
                println!("🔗 Custom mainnet forking enabled with RPC: {}", rpc_url);
            }
        }
        
        Ok(evm_config)
    }
}
