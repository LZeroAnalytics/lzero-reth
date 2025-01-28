use crate::primitives::alloy_primitives::{BlockNumber, StorageKey, StorageValue};
use alloy_primitives::{Address, Bytes, B256, U256};
use core::ops::{Deref, DerefMut};
use core::str::FromStr;
use reth_primitives::Account;
use reth_storage_errors::provider::{ProviderError, ProviderResult};
use revm::{
    db::DatabaseRef,
    primitives::{AccountInfo, Bytecode},
    Database,
};
use revm::primitives::hex;
use reqwest::blocking::Client;
use serde_json::json;
use serde::{Deserialize, Serialize};
use once_cell::sync::Lazy;

// Define your RPC endpoint
const MAINNET_RPC_URL: &str = "https://eth-mainnet.g.alchemy.com/v2/SHQeqTrXogQwqWL8veM2EOphUVoNw3mN";
static GLOBAL_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("Failed to build reqwest Client")
});

// JSON-RPC request and response structures
#[derive(Serialize)]
struct JsonRpcRequest<'a> {
    jsonrpc: &'a str,
    method: &'a str,
    params: serde_json::Value,
    id: u64,
}

#[derive(Deserialize)]
struct JsonRpcResponse<T> {
    jsonrpc: String,
    id: u64,
    result: Option<T>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

/// A helper trait responsible for providing state necessary for EVM execution.
///
/// This serves as the data layer for [`Database`].
pub trait EvmStateProvider: Send + Sync {
    /// Get basic account information.
    ///
    /// Returns [`None`] if the account doesn't exist.
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>>;

    /// Get the hash of the block with the given number. Returns [`None`] if no block with this
    /// number exists.
    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>>;

    /// Get account code by hash.
    fn bytecode_by_hash(
        &self,
        code_hash: &B256,
    ) -> ProviderResult<Option<reth_primitives::Bytecode>>;

    /// Get storage of the given account.
    fn storage(
        &self,
        account: Address,
        storage_key: StorageKey,
    ) -> ProviderResult<Option<StorageValue>>;
}

// Blanket implementation of EvmStateProvider for any type that implements StateProvider.
impl<T: reth_storage_api::StateProvider> EvmStateProvider for T {
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        <T as reth_storage_api::AccountReader>::basic_account(self, address)
    }

    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>> {
        <T as reth_storage_api::BlockHashReader>::block_hash(self, number)
    }

    fn bytecode_by_hash(
        &self,
        code_hash: &B256,
    ) -> ProviderResult<Option<reth_primitives::Bytecode>> {
        <T as reth_storage_api::StateProvider>::bytecode_by_hash(self, code_hash)
    }

    fn storage(
        &self,
        account: Address,
        storage_key: StorageKey,
    ) -> ProviderResult<Option<StorageValue>> {
        <T as reth_storage_api::StateProvider>::storage(self, account, storage_key)
    }
}

/// A [Database] and [`DatabaseRef`] implementation that uses [`EvmStateProvider`] as the underlying
/// data source.
#[derive(Debug, Clone)]
pub struct StateProviderDatabase<DB>(pub DB);

impl<DB> StateProviderDatabase<DB> {
    /// Create new State with generic `StateProvider`.
    pub const fn new(db: DB) -> Self {
        Self(db)
    }

    /// Consume State and return inner `StateProvider`.
    pub fn into_inner(self) -> DB {
        self.0
    }

    // Generic JSON-RPC call
    fn rpc_call<T: for<'de> Deserialize<'de>>(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<T, String> {

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            method,
            params,
            id: 1,
        };

        let resp = GLOBAL_CLIENT
            .post(MAINNET_RPC_URL)
            .json(&request)
            .send()
            .map_err(|e| format!("HTTP request error: {}", e))?;

        let rpc_response: JsonRpcResponse<T> = resp
            .json()
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;

        if let Some(error) = rpc_response.error {
            return Err(format!("RPC error {}: {}", error.code, error.message));
        }

        rpc_response
            .result
            .ok_or_else(|| "No result in RPC response".to_string())
    }
}

impl<DB> AsRef<DB> for StateProviderDatabase<DB> {
    fn as_ref(&self) -> &DB {
        self
    }
}

impl<DB> Deref for StateProviderDatabase<DB> {
    type Target = DB;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<DB> DerefMut for StateProviderDatabase<DB> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<DB: EvmStateProvider> Database for StateProviderDatabase<DB> {
    type Error = ProviderError;

    /// Retrieves basic account information for a given address.
    ///
    /// Returns `Ok` with `Some(AccountInfo)` if the account exists,
    /// `None` if it doesn't, or an error if encountered.
    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        self.basic_ref(address)
    }

    /// Retrieves the bytecode associated with a given code hash.
    ///
    /// Returns `Ok` with the bytecode if found, or the default bytecode otherwise.
    fn code_by_hash(&mut self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        self.code_by_hash_ref(code_hash)
    }

    /// Retrieves the storage value at a specific index for a given address.
    ///
    /// Returns `Ok` with the storage value, or the default value if not found.
    fn storage(&mut self, address: Address, index: U256) -> Result<U256, Self::Error> {
        self.storage_ref(address, index)
    }

    /// Retrieves the block hash for a given block number.
    ///
    /// Returns `Ok` with the block hash if found, or the default hash otherwise.
    /// Note: It safely casts the `number` to `u64`.
    fn block_hash(&mut self, number: u64) -> Result<B256, Self::Error> {
        self.block_hash_ref(number)
    }
}

fn is_special_address(addr: &Address) -> bool {
    let sentinel1 = Address::from_str("0xfffffffffffffffffffffffffffffffffffffffe")
        .expect("valid sentinel address 1");
    let sentinel2 = Address::from_str("0x000f3df6d732807ef1319fb7b8bb8522d0beac02")
        .expect("valid sentinel address 2");

    *addr == sentinel1 || *addr == sentinel2
}

impl<DB: EvmStateProvider> DatabaseRef for StateProviderDatabase<DB> {
    type Error = <Self as Database>::Error;

    /// Retrieves basic account information for a given address.
    ///
    /// Returns `Ok` with `Some(AccountInfo)` if the account exists,
    /// `None` if it doesn't, or an error if encountered.
    fn basic_ref(&self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        if is_special_address(&address) {
            return Ok(self.basic_account(&address)?.map(Into::into));
        }

        if self.basic_account(&address)?.is_some() {
            println!("(LZero) - Found address {:?} in local storage", address);
            return Ok(self.basic_account(&address)?.map(Into::into));
        }

        println!("(LZero) - Retrieving data from mainnet for address: {:?}", address);
        let balance_hex: String = self
            .rpc_call("eth_getBalance", json!([address, "0x14B5D8C"]))
            .unwrap();
        let nonce_hex: String = self
            .rpc_call("eth_getTransactionCount", json!([address, "0x14B5D8C"]))
            .unwrap();
        let code_hex: String = self
            .rpc_call("eth_getCode", json!([address, "0x14B5D8C"]))
            .unwrap();

        let balance = U256::from_str_radix(balance_hex.trim_start_matches("0x"), 16)
            .map_err(|e| format!("Invalid balance hex: {}", e)).unwrap();
        let nonce = u64::from_str_radix(nonce_hex.trim_start_matches("0x"), 16)
            .map_err(|e| format!("Invalid nonce hex: {}", e)).unwrap();

        let code_bytes = hex::decode(code_hex.trim_start_matches("0x"))
            .map_err(|e| format!("Invalid code bytes hex: {}", e)).unwrap();

        let code_hash = crate::primitives::keccak256(&code_bytes);
        let code = if code_bytes.is_empty() {
            None
        } else {
            Some(Bytecode::new_raw(Bytes::from(code_bytes)))
        };

        let account_info = AccountInfo {
            balance,
            nonce,
            code_hash,
            code,
        };
        Ok(Some(account_info))
    }

    /// Retrieves the bytecode associated with a given code hash.
    ///
    /// Returns `Ok` with the bytecode if found, or the default bytecode otherwise.
    fn code_by_hash_ref(&self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        println!("(LZero) Querying bytecode by hash: {}", code_hash);
        Ok(self.bytecode_by_hash(&code_hash)?.unwrap_or_default().0)
    }

    /// Retrieves the storage value at a specific index for a given address.
    ///
    /// Returns `Ok` with the storage value, or the default value if not found.
    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        if is_special_address(&address) {
            return Ok(self.0.storage(address, B256::new(index.to_be_bytes()))?.unwrap_or_default());
        }

        println!(
            "(LZero) Querying storage for account: {:?} and storage key: {}",
            address, index
        );

        let local_val = self.0.storage(address, B256::new(index.to_be_bytes()))?.unwrap_or_default();

        if local_val != U256::ZERO {
            println!("(LZero) Found storage value in local storage for address: {:?}", address);
            return Ok(local_val.into());
        }

        println!("(LZero) Retrieving mainnet storage slot for address: {:?}", address);
        let index_hex = format!("0x{:x}", index);
        let storage_hex: String = self
            .rpc_call("eth_getStorageAt", json!([address, index_hex, "0x14B5D8C"]))
            .unwrap();

        let storage_u256 = U256::from_str_radix(storage_hex.trim_start_matches("0x"), 16)
            .map_err(|e| format!("Invalid storage hex: {}", e)).unwrap();

        Ok(storage_u256)
    }

    /// Retrieves the block hash for a given block number.
    ///
    /// Returns `Ok` with the block hash if found, or the default hash otherwise.
    fn block_hash_ref(&self, number: u64) -> Result<B256, Self::Error> {
        println!("(LZero) Querying block hash for number: {}", number);
        Ok(self.0.block_hash(number)?.unwrap_or_default())
    }
}