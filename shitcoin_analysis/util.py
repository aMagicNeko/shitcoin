from solana.rpc.types import TokenAccountOpts, TxOpts, MemcmpOpts
import solders.system_program as system_program
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price #type: ignore
from solders.keypair import Keypair #type: ignore
from solders.transaction import VersionedTransaction #type: ignore
from solana.transaction import AccountMeta
from solders.message import MessageV0 #type: ignore
from solders.instruction import Instruction  # type: ignore
from spl.token.client import Token
from spl.token.constants import WRAPPED_SOL_MINT
from spl.token.instructions import close_account, CloseAccountParams, create_associated_token_account, get_associated_token_address
import spl.token.instructions as spl_token_instructions
from solders.pubkey import Pubkey #type: ignore
import json
import time
import threading
import requests
from construct import Bytes, Int32ul, Int8ul, Int64ul, Padding, BitsInteger, BitsSwapped, BitStruct, Const, Flag, BytesInteger
from construct import Struct as cStruct
from solana.rpc.api import Client
from solders.keypair import Keypair #type: ignore

PUB_KEY = "BAwzi681zGP8WMT57piNvm5dPMLdDRrQT1L6YMqZomoK"
PRIV_KEY = "48iiNa3w5n3XSQ9L6dQK6d2hvjXfZZQGgnDTpTMrRaA69ky6ZbijPAZ1NJyW7PqFgj9LPcNWHv6vCeFNrP4DcMLh"
#RPC = "https://api.mainnet-beta.solana.com/"
RPC = "https://solana-mainnet.api.syndica.io/api-token/2Tq515vn1usWZBd1CED2ryeAjZsbAfDrwp14W9fyxpUgoAA8J9atUgCzEgNATAxGm4mYgaCrcGhvXUqhdnzvu29j7JDwTLV59Nc6R83wAtHub1MMLyRC9eGXMrK4Z5EDAYvhr1nun27e1G6uEABbjhSKHAhGXapKhrQZWyH4s4ZhpBFEBCnpp6bmGctnsPAsBcTCyH4vwYhxhBU788AVqTJYJFShudP3MTSzmEazaK8CSR11fs5kBtGK7kvJDeeBLuvJM7coBoG41KQMG6pR4gPZgL3KrxJSPerRSLhYUTEC9KGhptkxhL71Ym9qtwQ2XkuMtdjmvaUGpTbUXbdjggqrmsiYh4aJacDfxAoWKrre5TWfDmvqRwqYJCL8wh3Pzom2oUnLDjM5q85husW2pAx57EmWNfyd2PbGoCSRpateLmBWbpNZ29tJuEFHEGV7R4gwZG5iRRm9XtwotsavtR4wCPk7HiwxN2CFn3YopEZk7L4jVgJYfjvCQMQr7"
client = Client(RPC)
WSOL = Pubkey.from_string("So11111111111111111111111111111111111111112")
RAY_V4 = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
RAY_AUTHORITY_V4 = Pubkey.from_string("5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1")
OPEN_BOOK_PROGRAM = Pubkey.from_string("srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
SOL = "So11111111111111111111111111111111111111112"

LAMPORTS_PER_SOL = 1_000_000_000
UNIT_PRICE =  250_000
UNIT_BUDGET =  80_000
# NOT MY WORK, THANK YOU TO WHOEVER FIGURED THIS OUT

LIQUIDITY_STATE_LAYOUT_V4 = cStruct(
    "status" / Int64ul,
    "nonce" / Int64ul,
    "orderNum" / Int64ul,
    "depth" / Int64ul,
    "coinDecimals" / Int64ul,
    "pcDecimals" / Int64ul,
    "state" / Int64ul,
    "resetFlag" / Int64ul,
    "minSize" / Int64ul,
    "volMaxCutRatio" / Int64ul,
    "amountWaveRatio" / Int64ul,
    "coinLotSize" / Int64ul,
    "pcLotSize" / Int64ul,
    "minPriceMultiplier" / Int64ul,
    "maxPriceMultiplier" / Int64ul,
    "systemDecimalsValue" / Int64ul,
    "minSeparateNumerator" / Int64ul,
    "minSeparateDenominator" / Int64ul,
    "tradeFeeNumerator" / Int64ul,
    "tradeFeeDenominator" / Int64ul,
    "pnlNumerator" / Int64ul,
    "pnlDenominator" / Int64ul,
    "swapFeeNumerator" / Int64ul,
    "swapFeeDenominator" / Int64ul,
    "needTakePnlCoin" / Int64ul,
    "needTakePnlPc" / Int64ul,
    "totalPnlPc" / Int64ul,
    "totalPnlCoin" / Int64ul,
    "poolOpenTime" / Int64ul,
    "punishPcAmount" / Int64ul,
    "punishCoinAmount" / Int64ul,
    "orderbookToInitTime" / Int64ul,
    "swapCoinInAmount" / BytesInteger(16, signed=False, swapped=True),
    "swapPcOutAmount" / BytesInteger(16, signed=False, swapped=True),
    "swapCoin2PcFee" / Int64ul,
    "swapPcInAmount" / BytesInteger(16, signed=False, swapped=True),
    "swapCoinOutAmount" / BytesInteger(16, signed=False, swapped=True),
    "swapPc2CoinFee" / Int64ul,
    "poolCoinTokenAccount" / Bytes(32),
    "poolPcTokenAccount" / Bytes(32),
    "coinMintAddress" / Bytes(32),
    "pcMintAddress" / Bytes(32),
    "lpMintAddress" / Bytes(32),
    "ammOpenOrders" / Bytes(32),
    "serumMarket" / Bytes(32),
    "serumProgramId" / Bytes(32),
    "ammTargetOrders" / Bytes(32),
    "poolWithdrawQueue" / Bytes(32),
    "poolTempLpTokenAccount" / Bytes(32),
    "ammOwner" / Bytes(32),
    "pnlOwner" / Bytes(32),
)

ACCOUNT_FLAGS_LAYOUT = BitsSwapped(  
    BitStruct(
        "initialized" / Flag,
        "market" / Flag,
        "open_orders" / Flag,
        "request_queue" / Flag,
        "event_queue" / Flag,
        "bids" / Flag,
        "asks" / Flag,
        Const(0, BitsInteger(57)),
    )
)

MARKET_STATE_LAYOUT_V3 = cStruct(
    Padding(5),
    "account_flags" / ACCOUNT_FLAGS_LAYOUT,
    "own_address" / Bytes(32),
    "vault_signer_nonce" / Int64ul,
    "base_mint" / Bytes(32),
    "quote_mint" / Bytes(32),
    "base_vault" / Bytes(32),
    "base_deposits_total" / Int64ul,
    "base_fees_accrued" / Int64ul,
    "quote_vault" / Bytes(32),
    "quote_deposits_total" / Int64ul,
    "quote_fees_accrued" / Int64ul,
    "quote_dust_threshold" / Int64ul,
    "request_queue" / Bytes(32),
    "event_queue" / Bytes(32),
    "bids" / Bytes(32),
    "asks" / Bytes(32),
    "base_lot_size" / Int64ul,
    "quote_lot_size" / Int64ul,
    "fee_rate_bps" / Int64ul,
    "referrer_rebate_accrued" / Int64ul,
    Padding(7),
)

OPEN_ORDERS_LAYOUT = cStruct(
    Padding(5),
    "account_flags" / ACCOUNT_FLAGS_LAYOUT,
    "market" / Bytes(32),
    "owner" / Bytes(32),
    "base_token_free" / Int64ul,
    "base_token_total" / Int64ul,
    "quote_token_free" / Int64ul,
    "quote_token_total" / Int64ul,
    "free_slot_bits" / Bytes(16),
    "is_bid_bits" / Bytes(16),
    "orders" / Bytes(16)[128],
    "client_ids" / Int64ul[128],
    "referrer_rebate_accrued" / Int64ul,
    Padding(7),
)

SWAP_LAYOUT = cStruct(
    "instruction" / Int8ul, "amount_in" / Int64ul, "min_amount_out" / Int64ul
)

PUBLIC_KEY_LAYOUT = Bytes(32)

ACCOUNT_LAYOUT = cStruct(
    "mint" / PUBLIC_KEY_LAYOUT,
    "owner" / PUBLIC_KEY_LAYOUT,
    "amount" / Int64ul,
    "delegate_option" / Int32ul,
    "delegate" / PUBLIC_KEY_LAYOUT,
    "state" / Int8ul,
    "is_native_option" / Int32ul,
    "is_native" / Int64ul,
    "delegated_amount" / Int64ul,
    "close_authority_option" / Int32ul,
    "close_authority" / PUBLIC_KEY_LAYOUT,
)


POOL_INFO_LAYOUT = cStruct(
    "instruction" / Int8ul,
    "simulate_type" / Int8ul
)

def make_swap_instruction(amount_in: int, token_account_in: Pubkey, token_account_out: Pubkey, accounts: dict, owner: Pubkey) -> Instruction:
    try:
        keys = [
            AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts["amm_id"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts["authority"], is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts["open_orders"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts["target_orders"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts["base_vault"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts["quote_vault"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=OPEN_BOOK_PROGRAM, is_signer=False, is_writable=False), 
            AccountMeta(pubkey=accounts["market_id"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts["bids"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts["asks"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts["event_queue"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts["market_base_vault"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts["market_quote_vault"], is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts["market_authority"], is_signer=False, is_writable=False),
            AccountMeta(pubkey=token_account_in, is_signer=False, is_writable=True),  
            AccountMeta(pubkey=token_account_out, is_signer=False, is_writable=True), 
            AccountMeta(pubkey=owner.pubkey(), is_signer=True, is_writable=False) 
        ]
        
        data = SWAP_LAYOUT.build(
            dict(
                instruction=9,
                amount_in=int(amount_in),
                min_amount_out=0 # TODO: slipperage
            )
        )
        return Instruction(RAY_V4, data, keys)
    except:
        return None

def get_token_account(owner: Pubkey, mint: Pubkey):
    try:
        account_data = client.get_token_accounts_by_owner(owner, TokenAccountOpts(mint))
        token_account = account_data.value[0].pubkey
        token_account_instructions = None
        return token_account, token_account_instructions
    except:
        token_account = get_associated_token_address(owner, mint)
        token_account_instructions = create_associated_token_account(owner, owner, mint)
        return token_account, token_account_instructions

pools_keys_dict = {}
def fetch_pool_keys(pair_address: str) -> dict:
    if pair_address in pools_keys_dict:
        return pools_keys_dict[pair_address]
    failed_cnt = 0
    while True:
        try:
            amm_id = Pubkey.from_string(pair_address)
            amm_data = client.get_account_info_json_parsed(amm_id).value.data
            amm_data_decoded = LIQUIDITY_STATE_LAYOUT_V4.parse(amm_data)
            OPEN_BOOK_PROGRAM = Pubkey.from_bytes(amm_data_decoded.serumProgramId)
            marketId = Pubkey.from_bytes(amm_data_decoded.serumMarket)
            marketInfo = client.get_account_info_json_parsed(marketId).value.data
            market_decoded = MARKET_STATE_LAYOUT_V3.parse(marketInfo)

            pool_keys = {
                "amm_id": amm_id,
                "base_mint": Pubkey.from_bytes(market_decoded.base_mint),
                "quote_mint": Pubkey.from_bytes(market_decoded.quote_mint),
                "lp_mint": Pubkey.from_bytes(amm_data_decoded.lpMintAddress),
                "version": 4,
                "base_decimals": amm_data_decoded.coinDecimals,
                "quote_decimals": amm_data_decoded.pcDecimals,
                "lpDecimals": amm_data_decoded.coinDecimals,
                "programId": RAY_V4,
                "authority": RAY_AUTHORITY_V4,
                "open_orders": Pubkey.from_bytes(amm_data_decoded.ammOpenOrders),
                "target_orders": Pubkey.from_bytes(amm_data_decoded.ammTargetOrders),
                "base_vault": Pubkey.from_bytes(amm_data_decoded.poolCoinTokenAccount),
                "quote_vault": Pubkey.from_bytes(amm_data_decoded.poolPcTokenAccount),
                "withdrawQueue": Pubkey.from_bytes(amm_data_decoded.poolWithdrawQueue),
                "lpVault": Pubkey.from_bytes(amm_data_decoded.poolTempLpTokenAccount),
                "marketProgramId": OPEN_BOOK_PROGRAM,
                "market_id": marketId,
                "market_authority": Pubkey.create_program_address(
                    [bytes(marketId)]
                    + [bytes([market_decoded.vault_signer_nonce])]
                    + [bytes(7)],
                    OPEN_BOOK_PROGRAM,
                ),
                "market_base_vault": Pubkey.from_bytes(market_decoded.base_vault),
                "market_quote_vault": Pubkey.from_bytes(market_decoded.quote_vault),
                "bids": Pubkey.from_bytes(market_decoded.bids),
                "asks": Pubkey.from_bytes(market_decoded.asks),
                "event_queue": Pubkey.from_bytes(market_decoded.event_queue),
                "pool_open_time": amm_data_decoded.poolOpenTime
            }
            pools_keys_dict[pair_address] = pool_keys
            return pool_keys
        except:
            failed_cnt += 1
            if failed_cnt >= 10:
                return None
    
def find_data(data: dict, field: str) -> str:
    if isinstance(data, dict):
        if field in data:
            return data[field]
        else:
            for value in data.values():
                result = find_data(value, field)
                if result is not None:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = find_data(item, field)
            if result is not None:
                return result
    return None

def get_token_balance(token_address: str) -> float:
    try:

        headers = {"accept": "application/json", "content-type": "application/json"}

        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "getTokenAccountsByOwner",
            "params": [
                PUB_KEY,
                {"mint": token_address},
                {"encoding": "jsonParsed"},
            ],
        }
        
        response = requests.post(RPC, json=payload, headers=headers)
        ui_amount = find_data(response.json(), "uiAmount")
        return float(ui_amount)
    except:
        return None

def get_token_balance_lamports(token_address: str) -> int:
    try:

        headers = {"accept": "application/json", "content-type": "application/json"}

        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "getTokenAccountsByOwner",
            "params": [
                PUB_KEY,
                {"mint": token_address},
                {"encoding": "jsonParsed"},
            ],
        }
        
        response = requests.post(RPC, json=payload, headers=headers)
        amount = find_data(response.json(), "amount")
        return int(amount)
    except:
        return None

def confirm_txn(txn_sig, max_retries: int = 20, retry_interval: int = 3) -> bool:
    retries = 0
    
    while retries < max_retries:
        try:
            txn_res = client.get_transaction(txn_sig, encoding="json", commitment="confirmed", max_supported_transaction_version=0)
            txn_json = json.loads(txn_res.value.transaction.meta.to_json())
            
            if txn_json['err'] is None:
                print("Transaction confirmed... try count:", retries)
                return True
            
            print("Error: Transaction not confirmed. Retrying...")
            if txn_json['err']:
                print("Transaction failed.")
                return False
        except Exception as e:
            print("Awaiting confirmation... try count:", retries)
            retries += 1
            time.sleep(retry_interval)
    
    print("Max retries reached. Transaction confirmation failed.")
    return None

def get_pair_address_from_rpc(token_address: str) -> str:
    BASE_OFFSET = 400
    QUOTE_OFFSET = 432
    DATA_LENGTH_FILTER = 752
    QUOTE_MINT = "So11111111111111111111111111111111111111112"
    RAYDIUM_PROGRAM_ID = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
    
    def fetch_amm_id(base_mint: str, quote_mint: str) -> str:
        memcmp_filter_base = MemcmpOpts(offset=BASE_OFFSET, bytes=base_mint)
        memcmp_filter_quote = MemcmpOpts(offset=QUOTE_OFFSET, bytes=quote_mint)
        try:
            response = client.get_program_accounts(
                RAYDIUM_PROGRAM_ID, 
                filters=[DATA_LENGTH_FILTER, memcmp_filter_base, memcmp_filter_quote]
            )
            accounts = response.value
            if accounts:
                return str(accounts[0].pubkey)
        except Exception as e:
            print(f"Error fetching AMM ID: {e}")
        return None

    # First attempt: base_mint at BASE_OFFSET, QUOTE_MINT at QUOTE_OFFSET
    pair_address = fetch_amm_id(token_address, QUOTE_MINT)
    
    # Second attempt: QUOTE_MINT at BASE_OFFSET, base_mint at QUOTE_OFFSET
    if not pair_address:
        pair_address = fetch_amm_id(QUOTE_MINT, token_address)
    
    return pair_address

def get_pair_address(token_address) -> str:
    url = f"https://api.dexscreener.com/latest/dex/search?q={token_address}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['pairs'][0]['pairAddress']
    else:
        return None

def get_token_price(pair_address: str) -> float:
    try:
        # Get AMM data and parse
        amm_pubkey = Pubkey.from_string(pair_address)
        amm_data = client.get_account_info(amm_pubkey).value.data
        liquidity_state = LIQUIDITY_STATE_LAYOUT_V4.parse(amm_data)
        
        # Extract relevant attributes with improved names
        amm_open_orders_pubkey = liquidity_state.ammOpenOrders
        pool_coin_token_account_pubkey = liquidity_state.poolCoinTokenAccount
        pool_pc_token_account_pubkey = liquidity_state.poolPcTokenAccount
        coin_decimals = liquidity_state.coinDecimals
        pc_decimals = liquidity_state.pcDecimals
        coin_mint_address = liquidity_state.coinMintAddress
         
        need_take_pnl_coin = liquidity_state.needTakePnlCoin
        need_take_pnl_pc = liquidity_state.needTakePnlPc

        # Create a dictionary to store fetched data
        fetched_data = {}
        try:
            # Create threads for fetching account info and token balances
            account_info_thread = threading.Thread(target=lambda: fetched_data.update({'open_orders_data': client.get_account_info(Pubkey.from_bytes(amm_open_orders_pubkey)).value.data}))
            base_balance_thread = threading.Thread(target=lambda: fetched_data.update({pool_coin_token_account_pubkey: client.get_token_account_balance(Pubkey.from_bytes(pool_coin_token_account_pubkey)).value.ui_amount}))
            quote_balance_thread = threading.Thread(target=lambda: fetched_data.update({pool_pc_token_account_pubkey: client.get_token_account_balance(Pubkey.from_bytes(pool_pc_token_account_pubkey)).value.ui_amount}))

            # Start threads
            account_info_thread.start()
            base_balance_thread.start()
            quote_balance_thread.start()

            # Wait for threads to complete
            account_info_thread.join()
            base_balance_thread.join()
            quote_balance_thread.join()
        except:
            return

        # Parse fetched data
        open_orders_data = fetched_data['open_orders_data']
        open_orders = OPEN_ORDERS_LAYOUT.parse(open_orders_data)
        base_token_total = open_orders.base_token_total
        quote_token_total = open_orders.quote_token_total
        
        # Get decimal factors
        base_decimal = 10 ** coin_decimals
        quote_decimal = 10 ** pc_decimals

        # Calculate PnL
        base_pnl = need_take_pnl_coin / base_decimal
        quote_pnl = need_take_pnl_pc / quote_decimal

        # Calculate token totals from open orders
        open_orders_base_token_total = base_token_total / base_decimal
        open_orders_quote_token_total = quote_token_total/ quote_decimal

        # Get token balances from fetched data
        base_token_amount = fetched_data[pool_coin_token_account_pubkey]
        quote_token_amount = fetched_data[pool_pc_token_account_pubkey]

        # Calculate total token amounts
        base = (base_token_amount or 0) + open_orders_base_token_total - base_pnl
        quote = (quote_token_amount or 0) + open_orders_quote_token_total - quote_pnl

        # Determine price in SOL
        price_in_sol = 0
        if Pubkey.from_bytes(coin_mint_address) == WSOL:
            price_in_sol = str(base / quote)
        else:
            price_in_sol = str(quote / base)

        return float(price_in_sol)

    except:
        return None

def buy(pair_address: str, amount_in_sol: float, pool_keys = None, simulate=False):

    # Fetch pool keys
    print("Fetching pool keys...")
    if pool_keys is None:
        pool_keys = fetch_pool_keys(pair_address)
    
    # Check if pool keys exist
    if pool_keys is None:
        print("No pools keys found...")
        return None

    # Determine the mint based on pool keys
    mint = pool_keys['base_mint'] if str(pool_keys['base_mint']) != SOL else pool_keys['quote_mint']
    amount_in = int(amount_in_sol * LAMPORTS_PER_SOL)

    # Get token account and token account instructions
    print("Getting token account...")
    token_account, token_account_instructions = get_token_account(payer_keypair.pubkey(), mint)

    # Get minimum balance needed for token account
    print("Getting minimum balance for token account...")
    balance_needed = Token.get_min_balance_rent_for_exempt_for_account(client)

    # Create a keypair for wrapped SOL (wSOL)
    print("Creating keypair for wSOL...")
    wsol_account_keypair = Keypair()
    wsol_token_account = wsol_account_keypair.pubkey()
    
    instructions = []

    # Create instructions to create a wSOL account, include the amount in 
    print("Creating wSOL account instructions...")
    create_wsol_account_instructions = system_program.create_account(
        system_program.CreateAccountParams(
            from_pubkey=payer_keypair.pubkey(),
            to_pubkey=wsol_account_keypair.pubkey(),
            lamports=int(balance_needed + amount_in),
            space=ACCOUNT_LAYOUT.sizeof(),
            owner=TOKEN_PROGRAM_ID,
        )
    )

    # Initialize wSOL account
    print("Initializing wSOL account...")
    init_wsol_account_instructions = spl_token_instructions.initialize_account(
        spl_token_instructions.InitializeAccountParams(
            account=wsol_account_keypair.pubkey(),
            mint=WSOL,
            owner=payer_keypair.pubkey(),
            program_id=TOKEN_PROGRAM_ID,
        )
    )

    # Create swap instructions
    print("Creating swap instructions...")
    swap_instructions = make_swap_instruction(amount_in, wsol_token_account, token_account, pool_keys, payer_keypair)

    # Create close account instructions for wSOL account
    print("Creating close account instructions...")
    close_account_instructions = close_account(CloseAccountParams(TOKEN_PROGRAM_ID, wsol_token_account, payer_keypair.pubkey(), payer_keypair.pubkey()))

    # Append instructions to the list
    print("Appending instructions...")
    instructions.append(set_compute_unit_limit(UNIT_BUDGET)) 
    instructions.append(set_compute_unit_price(UNIT_PRICE))
    instructions.append(create_wsol_account_instructions)
    instructions.append(init_wsol_account_instructions)
    if token_account_instructions:
        instructions.append(token_account_instructions)
    instructions.append(swap_instructions)
    instructions.append(close_account_instructions)

    # Compile the message
    print("Compiling message...")
    compiled_message = MessageV0.try_compile(
        payer_keypair.pubkey(),
        instructions,
        [],  
        client.get_latest_blockhash().value.blockhash,
    )
    if simulate:
        res = client.simulate_transaction(transaction, True)
        print("simulate res: ", res.to_json())
    # Create and send transaction
    print("Creating and sending transaction...")
    transaction = VersionedTransaction(compiled_message, [payer_keypair, wsol_account_keypair])
    txn_sig = client.send_transaction(transaction, opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed")).value
    print("Transaction Signature:", txn_sig)
    
    # Confirm transaction
    print("Confirming transaction...")
    confirm = confirm_txn(txn_sig)
    print(confirm)

def sell(pair_address: str, amount_in_lamports: int, pool_keys = None):

    # Convert amount to integer
    amount_in = int(amount_in_lamports)
    
    # Fetch pool keys
    print("Fetching pool keys...")
    if pool_keys is None:
        pool_keys = fetch_pool_keys(pair_address)
    
    # Check if pool keys exist
    if pool_keys is None:
        print("No pools keys found...")
        return None
        
    # Determine the mint based on pool keys
    mint = pool_keys['base_mint'] if str(pool_keys['base_mint']) != SOL else pool_keys['quote_mint']
    
    # Get token account
    print("Getting token account...")
    token_account = client.get_token_accounts_by_owner(payer_keypair.pubkey(), TokenAccountOpts(mint)).value[0].pubkey
    
    # Get wSOL token account and instructions
    print("Getting wSOL token account...")
    wsol_token_account, wsol_token_account_instructions = get_token_account(payer_keypair.pubkey(), WRAPPED_SOL_MINT)
    
    # Create swap instructions
    print("Creating swap instructions...")
    swap_instructions = make_swap_instruction(amount_in, token_account, wsol_token_account, pool_keys, payer_keypair)
    
    # Create close account instructions for wSOL account
    print("Creating close account instructions...")
    close_account_instructions = close_account(CloseAccountParams(TOKEN_PROGRAM_ID, wsol_token_account, payer_keypair.pubkey(), payer_keypair.pubkey()))

    # Initialize instructions list
    instructions = []
    print("Appending instructions...")
    instructions.append(set_compute_unit_limit(UNIT_BUDGET)) 
    instructions.append(set_compute_unit_price(UNIT_PRICE))
    if wsol_token_account_instructions:
        instructions.append(wsol_token_account_instructions)
    instructions.append(swap_instructions)
    instructions.append(close_account_instructions)
    
    # Compile the message
    print("Compiling message...")
    compiled_message = MessageV0.try_compile(
        payer_keypair.pubkey(),
        instructions,
        [],  
        client.get_latest_blockhash().value.blockhash,
    )

    # Create and send transaction
    print("Creating and sending transaction...")
    transaction = VersionedTransaction(compiled_message, [payer_keypair])
    txn_sig = client.send_transaction(transaction, opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed")).value
    print("Transaction Signature:", txn_sig)
    
    # Confirm transaction
    print("Confirming transaction...")
    confirm = confirm_txn(txn_sig)
    print(confirm)

def make_simulate_pool_info_instruction(accounts, mint, ctx):
    keys = [
        AccountMeta(pubkey=accounts["amm_id"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["authority"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["open_orders"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["base_vault"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["quote_vault"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["lp_mint"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["market_id"], is_signer=False, is_writable=False),    
        AccountMeta(pubkey=accounts['event_queue'], is_signer=False, is_writable=False),    
    ]
    data = POOL_INFO_LAYOUT.build(
        dict(
            instruction=12,
            simulate_type=0
        )
    )
    return Instruction(RAY_V4, data, keys)

def swap_token_amount_base_in(amount_in, total_coin_without_take_pnl, total_pc_without_take_pnl, zero_for_one):
    amount_in = amount_in * (10000 - 25) / 10000
    amount_out = None
    if zero_for_one:
        denominator = total_coin_without_take_pnl + amount_in
        amount_out = (total_pc_without_take_pnl * amount_in) // denominator
    else:
        denominator = total_pc_without_take_pnl + amount_in
        amount_out = (total_coin_without_take_pnl * amount_in) // denominator
    return int(amount_out)