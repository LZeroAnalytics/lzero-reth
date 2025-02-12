use crate::primitives::alloy_primitives::{BlockNumber, StorageKey, StorageValue};
use alloy_primitives::{Address, Bytes, B256, U256};
use core::ops::{Deref, DerefMut};
use core::str::FromStr;
use std::collections::HashMap;
use std::thread::sleep;
use std::time::{Duration};
use rand::Rng;
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
use moka::sync::Cache;
use reth_tracing::tracing::{debug, warn};

#[derive(Clone)]
pub struct OverrideAccount {
    /// The account balance.
    pub balance: U256,
    /// The account nonce.
    pub nonce: u64,
    /// Optional EVM bytecode.
    pub code: Option<Bytecode>,
    /// A mapping from storage slot (as U256) to the stored value.
    pub storage: HashMap<U256, U256>,
}

/// The full override accounts mapping for our special addresses.
pub static OVERRIDE_ACCOUNTS: Lazy<HashMap<Address, OverrideAccount>> = Lazy::new(|| {
    let mut accounts = HashMap::new();

    let addr1 = Address::from_str("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
        .expect("Valid address for override account 1");
    let addr2 = Address::from_str("0xdac17f958d2ee523a2206206994597c13d831ec7")
        .expect("Valid address for override account 1");

    let code_hex = "0x608060405234801561000f575f5ffd5b50600436106100f0575f3560e01c806370a0823111610093578063a9059cbb11610063578063a9059cbb1461020b578063b68db3d11461021e578063dd62ed3e14610233578063f2fde38b1461026b575f5ffd5b806370a082311461019e5780638da5cb5b146101c657806395d89b41146101f0578063a457c2d7146101f8575f5ffd5b806318160ddd116100ce57806318160ddd1461014757806323b872dd14610159578063313ce5671461016c578063395093511461018b575f5ffd5b806306fdde03146100f4578063095ea7b314610112578063158ef93e14610135575b5f5ffd5b6100fc61027e565b60405161010991906108ea565b60405180910390f35b61012561012036600461093a565b61030a565b6040519015158152602001610109565b60065461012590610100900460ff1681565b6003545b604051908152602001610109565b610125610167366004610962565b610320565b6006546101799060ff1681565b60405160ff9091168152602001610109565b61012561019936600461093a565b610370565b61014b6101ac36600461099c565b6001600160a01b03165f9081526001602052604090205490565b5f546101d8906001600160a01b031681565b6040516001600160a01b039091168152602001610109565b6100fc6103a6565b61012561020636600461093a565b6103b3565b61012561021936600461093a565b6103e9565b61023161022c366004610a5b565b6103f5565b005b61014b610241366004610af0565b6001600160a01b039182165f90815260026020908152604080832093909416825291909152205490565b61023161027936600461099c565b61049a565b6004805461028b90610b21565b80601f01602080910402602001604051908101604052809291908181526020018280546102b790610b21565b80156103025780601f106102d957610100808354040283529160200191610302565b820191905f5260205f20905b8154815290600101906020018083116102e557829003601f168201915b505050505081565b5f61031633848461056a565b5060015b92915050565b5f61032c848484610677565b6001600160a01b0384165f90815260026020908152604080832033808552925290912054610366918691610361908690610b6d565b61056a565b5060019392505050565b335f8181526002602090815260408083206001600160a01b03871684529091528120549091610316918590610361908690610b80565b6005805461028b90610b21565b335f8181526002602090815260408083206001600160a01b03871684529091528120549091610316918590610361908690610b6d565b5f610316338484610677565b600654610100900460ff16156104485760405162461bcd60e51b8152602060048201526013602482015272105b1c9958591e481a5b9a5d1a585b1a5e9959606a1b60448201526064015b60405180910390fd5b610451816107c2565b600461045d8582610bdf565b50600561046a8482610bdf565b506006805460ff191660ff8416179055610484818661080e565b50506006805461ff001916610100179055505050565b5f546001600160a01b031633146104f35760405162461bcd60e51b815260206004820152601760248201527f43616c6c6572206973206e6f7420746865206f776e6572000000000000000000604482015260640161043f565b6001600160a01b0381166105495760405162461bcd60e51b815260206004820152601960248201527f4e6577206f776e6572206973207a65726f206164647265737300000000000000604482015260640161043f565b5f80546001600160a01b0319166001600160a01b0392909216919091179055565b6001600160a01b0383166105c05760405162461bcd60e51b815260206004820152601d60248201527f417070726f76652066726f6d20746865207a65726f2061646472657373000000604482015260640161043f565b6001600160a01b0382166106165760405162461bcd60e51b815260206004820152601b60248201527f417070726f766520746f20746865207a65726f20616464726573730000000000604482015260640161043f565b6001600160a01b038381165f8181526002602090815260408083209487168084529482529182902085905590518481527f8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b92591015b60405180910390a3505050565b6001600160a01b0383166106cd5760405162461bcd60e51b815260206004820152601e60248201527f5472616e736665722066726f6d20746865207a65726f20616464726573730000604482015260640161043f565b6001600160a01b0382166107235760405162461bcd60e51b815260206004820152601c60248201527f5472616e7366657220746f20746865207a65726f206164647265737300000000604482015260640161043f565b6001600160a01b0383165f908152600160205260408120805483929061074a908490610b6d565b90915550506001600160a01b0382165f9081526001602052604081208054839290610776908490610b80565b92505081905550816001600160a01b0316836001600160a01b03167fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef8360405161066a91815260200190565b5f546001600160a01b0316156105495760405162461bcd60e51b815260206004820152601160248201527013dddb995c88185b1c9958591e481cd95d607a1b604482015260640161043f565b6001600160a01b0382166108645760405162461bcd60e51b815260206004820152601860248201527f4d696e7420746f20746865207a65726f20616464726573730000000000000000604482015260640161043f565b8060035f8282546108759190610b80565b90915550506001600160a01b0382165f90815260016020526040812080548392906108a1908490610b80565b90915550506040518181526001600160a01b038316905f907fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef9060200160405180910390a35050565b602081525f82518060208401528060208501604085015e5f604082850101526040601f19601f83011684010191505092915050565b80356001600160a01b0381168114610935575f5ffd5b919050565b5f5f6040838503121561094b575f5ffd5b6109548361091f565b946020939093013593505050565b5f5f5f60608486031215610974575f5ffd5b61097d8461091f565b925061098b6020850161091f565b929592945050506040919091013590565b5f602082840312156109ac575f5ffd5b6109b58261091f565b9392505050565b634e487b7160e01b5f52604160045260245ffd5b5f82601f8301126109df575f5ffd5b813567ffffffffffffffff8111156109f9576109f96109bc565b604051601f8201601f19908116603f0116810167ffffffffffffffff81118282101715610a2857610a286109bc565b604052818152838201602001851015610a3f575f5ffd5b816020850160208301375f918101602001919091529392505050565b5f5f5f5f5f60a08688031215610a6f575f5ffd5b85359450602086013567ffffffffffffffff811115610a8c575f5ffd5b610a98888289016109d0565b945050604086013567ffffffffffffffff811115610ab4575f5ffd5b610ac0888289016109d0565b935050606086013560ff81168114610ad6575f5ffd5b9150610ae46080870161091f565b90509295509295909350565b5f5f60408385031215610b01575f5ffd5b610b0a8361091f565b9150610b186020840161091f565b90509250929050565b600181811c90821680610b3557607f821691505b602082108103610b5357634e487b7160e01b5f52602260045260245ffd5b50919050565b634e487b7160e01b5f52601160045260245ffd5b8181038181111561031a5761031a610b59565b8082018082111561031a5761031a610b59565b601f821115610bda57805f5260205f20601f840160051c81016020851015610bb85750805b601f840160051c820191505b81811015610bd7575f8155600101610bc4565b50505b505050565b815167ffffffffffffffff811115610bf957610bf96109bc565b610c0d81610c078454610b21565b84610b93565b6020601f821160018114610c3f575f8315610c285750848201515b5f19600385901b1c1916600184901b178455610bd7565b5f84815260208120601f198516915b82811015610c6e5787850151825560209485019460019092019101610c4e565b5084821015610c8b57868401515f19600387901b60f8161c191681555b50505050600190811b0190555056fea2646970667358221220b44bbe697b268583d0bf85e221d9190ad932ef7e5e11c9cafd24bc1fdad25d0564736f6c634300081c0033";
    let code_bytes = hex::decode(code_hex.trim_start_matches("0x"))
        .expect("Valid hex for override account");
    let code = if code_bytes.is_empty() {
        None
    } else {
        Some(Bytecode::new_raw(Bytes::from(code_bytes)))
    };

    let mut storage1 = HashMap::new();
    storage1.insert(
        U256::from_str("0x0000000000000000000000000000000000000000000000000000000000000000")
            .unwrap(),
        U256::from_str("0x8943545177806ed17b9f23f0a21ee5948ecaa776").unwrap(),
    );
    storage1.insert(
        U256::from_str("0x0000000000000000000000000000000000000000000000000000000000000003")
            .unwrap(),
        U256::from_str("0x0de0b6b3a7640000").unwrap(),
    );
    storage1.insert(
        U256::from_str("0x0000000000000000000000000000000000000000000000000000000000000004")
            .unwrap(),
        U256::from_str("0x55534420436f696e000000000000000000000000000000000000000000000010")
            .unwrap(),
    );
    storage1.insert(
        U256::from_str("0x0000000000000000000000000000000000000000000000000000000000000005")
            .unwrap(),
        U256::from_str("0x5553444300000000000000000000000000000000000000000000000000000008")
            .unwrap(),
    );
    storage1.insert(
        U256::from_str("0x0000000000000000000000000000000000000000000000000000000000000006")
            .unwrap(),
        U256::from_str("0x0106").unwrap(),
    );
    storage1.insert(
        U256::from_str("0x612e45a34980d9bb93dae5b3679ff2772e6cf57e473c7e5891ef9bc7f49f0f0e")
            .unwrap(),
        U256::from_str("0x0de0b6b3a7640000").unwrap(),
    );

    let mut storage2 = HashMap::new();
    storage2.insert(
        U256::from_str("0x0000000000000000000000000000000000000000000000000000000000000000")
            .unwrap(),
        U256::from_str("0x8943545177806ed17b9f23f0a21ee5948ecaa776").unwrap(),
    );
    storage2.insert(
        U256::from_str("0x0000000000000000000000000000000000000000000000000000000000000003")
            .unwrap(),
        U256::from_str("0x0de0b6b3a7640000").unwrap(),
    );
    storage2.insert(
        U256::from_str("0x0000000000000000000000000000000000000000000000000000000000000004")
            .unwrap(),
        U256::from_str("0x5465746865722055534400000000000000000000000000000000000000000014")
            .unwrap(),
    );
    storage2.insert(
        U256::from_str("0x0000000000000000000000000000000000000000000000000000000000000005")
            .unwrap(),
        U256::from_str("0x5553445400000000000000000000000000000000000000000000000000000008")
            .unwrap(),
    );
    storage2.insert(
        U256::from_str("0x0000000000000000000000000000000000000000000000000000000000000006")
            .unwrap(),
        U256::from_str("0x0106").unwrap(),
    );
    storage2.insert(
        U256::from_str("0x612e45a34980d9bb93dae5b3679ff2772e6cf57e473c7e5891ef9bc7f49f0f0e")
            .unwrap(),
        U256::from_str("0x0de0b6b3a7640000").unwrap(),
    );

    accounts.insert(
        addr1,
        OverrideAccount {
            balance: U256::ZERO,
            nonce: 0,
            code: code.clone(),
            storage: storage1
        }
    );

    accounts.insert(
        addr2,
        OverrideAccount {
            balance: U256::ZERO,
            nonce: 0,
            code: code.clone(),
            storage: storage2,
        },
    );

    accounts
});

// Cache for AccountInfo keyed by Address.
static ACCOUNT_CACHE: Lazy<Cache<Address, AccountInfo>> = Lazy::new(|| {
    Cache::builder()
        // Cache entries live for 5 minutes.
        .time_to_live(Duration::from_secs(300))
        .max_capacity(1000)
        .build()
});

// Cache for storage values keyed by (Address, U256).
static STORAGE_CACHE: Lazy<Cache<(Address, U256), U256>> = Lazy::new(|| {
    Cache::builder()
        .time_to_live(Duration::from_secs(300))
        .max_capacity(1000)
        .build()
});

// Define your RPC endpoint
fn get_rpc_url() -> String {
    std::env::var("FORKING_RPC_URL").unwrap_or_else(|_| {
        "".to_string()
    })
}

fn get_block_height() -> String {
    std::env::var("FORKING_BLOCK_HEIGHT").unwrap_or_else(|_| {
        "0x14B5D8C".to_string()
    })
}

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
        // Retry configuration
        const MAX_RETRIES: u32 = 5;
        const BASE_DELAY_MS: u64 = 100;
        const MAX_BACKOFF_MS: u64 = 1600; // Maximum backoff delay

        let mut attempt = 0;

        loop {
            let request = JsonRpcRequest {
                jsonrpc: "2.0",
                method,
                params: params.clone(), // Clone because we might retry
                id: 1,
            };

            let rpc_url = get_rpc_url();

            // Send the HTTP request
            let response = GLOBAL_CLIENT
                .post(rpc_url)
                .json(&request)
                .send();

            match response {
                Ok(resp) => {
                    // Attempt to parse the JSON-RPC response
                    let rpc_response: Result<JsonRpcResponse<T>, _> = resp.json();

                    match rpc_response {
                        Ok(rpc_resp) => {
                            if let Some(error) = rpc_resp.error {
                                // Decide whether to retry based on error code
                                if Self::is_transient_error(error.code) && attempt < MAX_RETRIES {
                                    attempt += 1;
                                    let delay = Self::calculate_backoff_delay(attempt, BASE_DELAY_MS, MAX_BACKOFF_MS);
                                    warn!(target: "LZero",
                                        error_code = error.code,
                                        error_message = %error.message,
                                        delay = ?delay,
                                        attempt,
                                        max_retries = MAX_RETRIES,
                                        "RPC error occurred, retrying...");
                                    sleep(delay);
                                    continue;
                                } else {
                                    // Non-retriable error or max retries reached
                                    return Err(format!("RPC error {}: {}", error.code, error.message));
                                }
                            }

                            // Check if the result is present
                            match rpc_resp.result {
                                Some(result) => return Ok(result),
                                None => {
                                    if attempt < MAX_RETRIES {
                                        attempt += 1;
                                        let delay = Self::calculate_backoff_delay(attempt, BASE_DELAY_MS, MAX_BACKOFF_MS);
                                        warn!(target: "LZero",
                                        delay = ?delay,
                                        attempt,
                                        max_retries = MAX_RETRIES,
                                        "No result in RPC response, retrying...");
                                        sleep(delay);
                                        continue;
                                    } else {
                                        return Err("No result in RPC response".to_string());
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            // JSON parsing error
                            if attempt < MAX_RETRIES {
                                attempt += 1;
                                let delay = Self::calculate_backoff_delay(attempt, BASE_DELAY_MS, MAX_BACKOFF_MS);
                                warn!(target: "LZero",
                                        error = ?e,
                                        delay = ?delay,
                                        attempt,
                                        max_retries = MAX_RETRIES,
                                        "Failed to parse RPC response, retrying...");
                                sleep(delay);
                                continue;
                            } else {
                                return Err(format!("Failed to parse JSON response: {}", e));
                            }
                        }
                    }
                }
                Err(e) => {
                    // Network or request error
                    if attempt < MAX_RETRIES {
                        attempt += 1;
                        let delay = Self::calculate_backoff_delay(attempt, BASE_DELAY_MS, MAX_BACKOFF_MS);
                        warn!(target: "LZero",
                                        error = ?e,
                                        delay = ?delay,
                                        attempt,
                                        max_retries = MAX_RETRIES,
                                        "HTTP RPC error occurred, retrying...");
                        sleep(delay);
                        continue;
                    } else {
                        return Err(format!("HTTP request error: {}", e));
                    }
                }
            }
        }
    }

    /// Determines if an RPC error code is transient and worth retrying
    fn is_transient_error(code: i64) -> bool {
        true
    }

    /// Calculates the backoff delay with exponential growth and jitter
    fn calculate_backoff_delay(attempt: u32, base_delay_ms: u64, max_delay_ms: u64) -> Duration {
        let exponential_delay = base_delay_ms * 2u64.pow(attempt - 1);
        let capped_delay = exponential_delay.min(max_delay_ms);
        // Add jitter: random value between 0 and 100ms
        let jitter = rand::thread_rng().gen_range(0..100);
        Duration::from_millis(capped_delay + jitter)
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

        if self.basic_account(&address)?.is_some() || get_rpc_url().is_empty() {
            return Ok(self.basic_account(&address)?.map(Into::into));
        }

        if let Some(override_acc) = OVERRIDE_ACCOUNTS.get(&address) {
            let account_info = AccountInfo {
                balance: override_acc.balance,
                nonce: override_acc.nonce,
                code_hash: override_acc.code.as_ref()
                    .map(|code| crate::primitives::keccak256(&code.bytes()))
                    .unwrap_or_default(),
                code: override_acc.code.clone(),
            };
            return Ok(Some(account_info));
        }

        if let Some(cached_account) = ACCOUNT_CACHE.get(&address) {
            return Ok(Some(cached_account));
        }

        let block_height = get_block_height();
        debug!(target: "LZero", ?address, "Retrieving account info from other network");
        let balance_hex: String = self
            .rpc_call("eth_getBalance", json!([address, block_height]))
            .unwrap();
        let nonce_hex: String = self
            .rpc_call("eth_getTransactionCount", json!([address, block_height]))
            .unwrap();
        let code_hex: String = self
            .rpc_call("eth_getCode", json!([address, block_height]))
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

        ACCOUNT_CACHE.insert(address, account_info.clone());
        Ok(Some(account_info))
    }

    /// Retrieves the bytecode associated with a given code hash.
    ///
    /// Returns `Ok` with the bytecode if found, or the default bytecode otherwise.
    fn code_by_hash_ref(&self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        Ok(self.bytecode_by_hash(&code_hash)?.unwrap_or_default().0)
    }

    /// Retrieves the storage value at a specific index for a given address.
    ///
    /// Returns `Ok` with the storage value, or the default value if not found.
    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        if is_special_address(&address) {
            return Ok(self.0.storage(address, B256::new(index.to_be_bytes()))?.unwrap_or_default());
        }

        let local_val = self.0.storage(address, B256::new(index.to_be_bytes()))?.unwrap_or_default();

        if local_val != U256::ZERO || get_rpc_url().is_empty() {
            return Ok(local_val.into());
        }

        if let Some(override_acc) = OVERRIDE_ACCOUNTS.get(&address) {
            if let Some(val) = override_acc.storage.get(&index) {
                return Ok(*val);
            } else {
                return Ok(U256::ZERO);
            }
        }

        if let Some(cached) = STORAGE_CACHE.get(&(address, index)) {
            return Ok(cached);
        }

        debug!(target: "LZero", ?address, ?index, "Retrieving storage slot from other network");
        let block_height = get_block_height();
        let index_hex = format!("0x{:x}", index);
        let storage_hex: String = self
            .rpc_call("eth_getStorageAt", json!([address, index_hex, block_height]))
            .unwrap();

        let storage_u256 = U256::from_str_radix(storage_hex.trim_start_matches("0x"), 16)
            .map_err(|e| format!("Invalid storage hex: {}", e)).unwrap();

        STORAGE_CACHE.insert((address, index), storage_u256);
        Ok(storage_u256)
    }

    /// Retrieves the block hash for a given block number.
    ///
    /// Returns `Ok` with the block hash if found, or the default hash otherwise.
    fn block_hash_ref(&self, number: u64) -> Result<B256, Self::Error> {
        Ok(self.0.block_hash(number)?.unwrap_or_default())
    }
}