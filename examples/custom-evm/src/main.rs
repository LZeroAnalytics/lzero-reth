//! This example shows how to implement a node with a custom EVM

#![cfg_attr(not(test), warn(unused_crate_dependencies))]
static ALLOC: reth_cli_util::allocator::Allocator = reth_cli_util::allocator::new_allocator();

use clap::{Args, Parser};
use reth::cli::Cli;
use reth_ethereum_cli::chainspec::EthereumChainSpecParser;
use alloy_consensus::Header;
use alloy_genesis::Genesis;
use alloy_primitives::{address, Address, Bytes};
use reth::{
    builder::{
        components::{ExecutorBuilder, PayloadServiceBuilder},
        BuilderContext, NodeBuilder,
    },
    payload::{EthBuiltPayload, EthPayloadBuilderAttributes},
    revm::{
        handler::register::EvmHandler,
        inspector_handle_register,
        precompile::{Precompile, PrecompileOutput, PrecompileSpecId},
        primitives::{CfgEnvWithHandlerCfg, Env, PrecompileResult, TxEnv},
        ContextPrecompiles, Database, Evm, EvmBuilder, GetInspector,
    },
    rpc::types::engine::PayloadAttributes,
    tasks::TaskManager,
    transaction_pool::{PoolTransaction, TransactionPool},
};
use reth_node_builder::{
    engine_tree_config::{
        TreeConfig, DEFAULT_MEMORY_BLOCK_BUFFER_TARGET, DEFAULT_PERSISTENCE_THRESHOLD,
    },
    EngineNodeLauncher,
};
use reth_provider::providers::BlockchainProvider2;
use reth_chainspec::{Chain, ChainSpec};
use reth_evm::env::EvmEnv;
use reth_evm_ethereum::EthEvmConfig;
use reth_node_api::{
    ConfigureEvm, ConfigureEvmEnv, FullNodeTypes, NextBlockEnvAttributes, NodeTypes,
    NodeTypesWithEngine, PayloadTypes,
};
use reth_node_core::{args::RpcServerArgs, node_config::NodeConfig};
use reth_node_ethereum::{
    node::{EthereumAddOns, EthereumPayloadBuilder},
    BasicBlockExecutorProvider, EthExecutionStrategyFactory, EthereumNode,
};
use reth_primitives::{EthPrimitives, TransactionSigned};
use reth_tracing::{RethTracer, Tracer};
use std::{convert::Infallible, sync::Arc};
use reth_tracing::tracing::warn;
use tracing::info;

#[derive(Debug, Clone, Args, PartialEq, Eq)]
#[command(next_help_heading = "Engine")]
pub struct EngineArgs {
    /// Enable the experimental engine features on reth binary
    ///
    /// DEPRECATED: experimental engine is default now, use --engine.legacy to enable the legacy
    /// functionality
    #[arg(long = "engine.experimental", default_value = "false")]
    pub experimental: bool,

    /// Enable the legacy engine on reth binary
    #[arg(long = "engine.legacy", default_value = "false")]
    pub legacy: bool,

    /// Configure persistence threshold for engine experimental.
    #[arg(long = "engine.persistence-threshold", conflicts_with = "legacy", default_value_t = DEFAULT_PERSISTENCE_THRESHOLD)]
    pub persistence_threshold: u64,

    /// Configure the target number of blocks to keep in memory.
    #[arg(long = "engine.memory-block-buffer-target", conflicts_with = "legacy", default_value_t = DEFAULT_MEMORY_BLOCK_BUFFER_TARGET)]
    pub memory_block_buffer_target: u64,
}

/// Custom EVM configuration
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MyEvmConfig {
    /// Wrapper around mainnet configuration
    inner: EthEvmConfig,
}

impl MyEvmConfig {
    pub const fn new(chain_spec: Arc<ChainSpec>) -> Self {
        Self { inner: EthEvmConfig::new(chain_spec) }
    }
}

impl MyEvmConfig {
    /// Sets the precompiles to the EVM handler
    ///
    /// This will be invoked when the EVM is created via [ConfigureEvm::evm] or
    /// [ConfigureEvm::evm_with_inspector]
    ///
    /// This will use the default mainnet precompiles and add additional precompiles.
    pub fn set_precompiles<EXT, DB>(handler: &mut EvmHandler<EXT, DB>)
    where
        DB: Database,
    {
        // first we need the evm spec id, which determines the precompiles
        let spec_id = handler.cfg.spec_id;

        // install the precompiles
        handler.pre_execution.load_precompiles = Arc::new(move || {
            let mut precompiles = ContextPrecompiles::new(PrecompileSpecId::from_spec_id(spec_id));
            precompiles.extend([(
                address!("0000000000000000000000000000000000000999"),
                Precompile::Env(Self::my_precompile).into(),
            )]);
            precompiles
        });
    }

    /// A custom precompile that does nothing
    fn my_precompile(_data: &Bytes, _gas: u64, _env: &Env) -> PrecompileResult {
        Ok(PrecompileOutput::new(0, Bytes::new()))
    }
}

impl ConfigureEvmEnv for MyEvmConfig {
    type Header = Header;
    type Transaction = TransactionSigned;

    type Error = Infallible;

    fn fill_tx_env(&self, tx_env: &mut TxEnv, transaction: &TransactionSigned, sender: Address) {
        self.inner.fill_tx_env(tx_env, transaction, sender);
    }

    fn fill_tx_env_system_contract_call(
        &self,
        env: &mut Env,
        caller: Address,
        contract: Address,
        data: Bytes,
    ) {
        self.inner.fill_tx_env_system_contract_call(env, caller, contract, data);
    }

    fn fill_cfg_env(&self, cfg_env: &mut CfgEnvWithHandlerCfg, header: &Self::Header) {
        self.inner.fill_cfg_env(cfg_env, header);
    }

    fn next_cfg_and_block_env(
        &self,
        parent: &Self::Header,
        attributes: NextBlockEnvAttributes,
    ) -> Result<EvmEnv, Self::Error> {
        self.inner.next_cfg_and_block_env(parent, attributes)
    }
}

impl ConfigureEvm for MyEvmConfig {
    type DefaultExternalContext<'a> = ();

    fn evm<DB: Database>(&self, db: DB) -> Evm<'_, Self::DefaultExternalContext<'_>, DB> {
        EvmBuilder::default()
            .with_db(db)
            // add additional precompiles
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
            // add additional precompiles
            .append_handler_register(MyEvmConfig::set_precompiles)
            .append_handler_register(inspector_handle_register)
            .build()
    }

    fn default_external_context<'a>(&self) -> Self::DefaultExternalContext<'a> {}
}

/// Builds a regular ethereum block executor that uses the custom EVM.
#[derive(Debug, Default, Clone, Copy)]
#[non_exhaustive]
pub struct MyExecutorBuilder;

impl<Node> ExecutorBuilder<Node> for MyExecutorBuilder
where
    Node: FullNodeTypes<Types: NodeTypes<ChainSpec = ChainSpec, Primitives = EthPrimitives>>,
{
    type EVM = MyEvmConfig;
    type Executor = BasicBlockExecutorProvider<EthExecutionStrategyFactory<Self::EVM>>;

    async fn build_evm(
        self,
        ctx: &BuilderContext<Node>,
    ) -> eyre::Result<(Self::EVM, Self::Executor)> {
        Ok((
            MyEvmConfig::new(ctx.chain_spec()),
            BasicBlockExecutorProvider::new(EthExecutionStrategyFactory::new(
                ctx.chain_spec(),
                MyEvmConfig::new(ctx.chain_spec()),
            )),
        ))
    }
}

/// Builds a regular ethereum block executor that uses the custom EVM.
#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct MyPayloadBuilder {
    inner: EthereumPayloadBuilder,
}

impl<Types, Node, Pool> PayloadServiceBuilder<Node, Pool> for MyPayloadBuilder
where
    Types: NodeTypesWithEngine<ChainSpec = ChainSpec, Primitives = EthPrimitives>,
    Node: FullNodeTypes<Types = Types>,
    Pool: TransactionPool<Transaction: PoolTransaction<Consensus = TransactionSigned>>
    + Unpin
    + 'static,
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

fn main() {
    // Standard Reth CLI setup
    reth_cli_util::sigsegv_handler::install();
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::env::set_var("RUST_BACKTRACE", "1");
    }

    // The "standard" Reth CLI entry point that captures *all* normal flags
    if let Err(err) = Cli::<EthereumChainSpecParser, EngineArgs>::parse().run(|builder, engine_args| async move {
        if engine_args.experimental {
            warn!(target: "reth::cli", "Experimental engine is default now, and the --engine.experimental flag is deprecated. To enable the legacy functionality, use --engine.legacy.");
        }

        // Choose whether to use legacy or the new engine
        let use_legacy_engine = engine_args.legacy;
        match use_legacy_engine {
            false => {
                // Build the standard `engine_tree_config` from CLI flags
                let engine_tree_config = TreeConfig::default()
                    .with_persistence_threshold(engine_args.persistence_threshold)
                    .with_memory_block_buffer_target(engine_args.memory_block_buffer_target);

                // **Here** is where you inject your custom executor/payload builder
                let handle = builder
                    .with_types_and_provider::<EthereumNode, BlockchainProvider2<_>>()
                    .with_components(
                        // Start with the default EthereumNode components
                        EthereumNode::components()
                            // Now inject your custom executor:
                            .executor(MyExecutorBuilder::default())
                            // and your custom payload builder:
                            .payload(MyPayloadBuilder::default())
                    )
                    // Everything else is still the standard Reth approach
                    .with_add_ons(EthereumAddOns::default())
                    .launch_with_fn(|builder| {
                        let launcher = EngineNodeLauncher::new(
                            builder.task_executor().clone(),
                            builder.config().datadir(),
                            engine_tree_config,
                        );
                        builder.launch_with(launcher)
                    })
                    .await?;

                handle.node_exit_future.await
            }
            true => {
                // Legacy engine path if user sets --engine.legacy
                info!(target: "reth::cli", "Running with legacy engine");
                let handle = builder.launch_node(EthereumNode::default()).await?;
                handle.node_exit_future.await
            }
        }
    }) {
        eprintln!("Error: {err:?}");
        std::process::exit(1);
    }
}