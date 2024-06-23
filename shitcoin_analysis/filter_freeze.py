import os
import shutil
import subprocess
import requests
def get_token(pair_address):
    url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{pair_address}"
    response = requests.get(url).json()
    print('here')
    if response['pairs'] is None:
        return 0
    for pair in response['pairs']:
        if pair['quoteToken']['address'] == 'So11111111111111111111111111111111111111112':
            return pair['baseToken']['address']
        elif pair['baseToken']['address'] == 'So11111111111111111111111111111111111111112':
            return pair['quoteToken']['address']
    

def get_all_parquet_files(root_dir):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.parquet'):
                file_paths.append(os.path.join(root, file))
    return file_paths

if __name__ == "__main__":
    source_dir = ""
    files = get_all_parquet_files(source_dir)
    for file in files:
        date = os.path.basename(os.path.dirname(file))
        token_address = get_token(os.path.basename(file).split('.')[0])
        command = f'spl-token display {token_address}'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            mint_authority = None
            freeze_authority = None
            for line in output_lines:
                if 'Freeze authority:' in line and "not set" in line:
                    output_dir = os.path.join("filted_data", date)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    shutil.move(file, output_dir)
                    print(f'Moved {file} to {output_dir}')
                    break
        else:
            print(f'Error checking authority for {token_address}: {result.stderr}')
    print('Done')
