import os
import shutil
import subprocess
import requests
from util import fetch_pool_keys, WSOL
def get_token(pair_address):
    url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{pair_address}"
    print(pair_address)
    response = requests.get(url).json()
    if response['pairs'] is None:
        return 0
    for pair in response['pairs']:
        if pair['quoteToken']['address'] == 'So11111111111111111111111111111111111111112':
            return pair['baseToken']['address']
        elif pair['baseToken']['address'] == 'So11111111111111111111111111111111111111112':
            return pair['quoteToken']['address']

def get_token1(pair_address):
    keys = fetch_pool_keys(pair_address)
    if keys['base_mint'] == WSOL:
        return keys['quote_mint'].__str__()
    else:
        return keys['base_mint'].__str__()
    
def get_all_parquet_files(root_dir):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.parquet'):
                file_paths.append(os.path.join(root, file))
    return file_paths

if __name__ == "__main__":
    source_dir =  ""
    files = get_all_parquet_files(source_dir)
    for file in files:
        try:
            date = os.path.basename(os.path.dirname(file))
            pair_address = os.path.basename(file).split('.')[0]
            token_address = get_token1(pair_address)
            if token_address == 0:
                token_address = pair_address
            print(token_address)
            fff = r'C:\Users\Administrator\solana-release\bin\spl-token.exe'
            command = f'{fff} display {token_address}'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                mint_authority = None
                freeze_authority = None
                for line in output_lines:
                    if 'Freeze authority:' in line and "not set" in line:
                        output_dir = os.path.join("filtered_data", date)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        shutil.move(file, output_dir)
                        print(f'Moved {file} to {output_dir}')
                        break
            else:
                print(f'Error checking authority for {token_address}: {result.stderr}')
        except Exception as e:
            print(f"failed to process {file}: {e}")
            continue
    print('Done')
