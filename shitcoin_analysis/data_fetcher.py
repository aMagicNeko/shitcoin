import requests
from bs4 import BeautifulSoup
import os
import shutil

# 服务器地址
base_url = 'http://43.134.79.111:40000/coin_data/'

# 创建下载目录
download_dir = 'downloads'
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# 目标目录
target_dir = '../coin_data'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

def download_file(url, dest_folder):
    local_filename = url.split('/')[-1]
    local_path = os.path.join(dest_folder, local_filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

def move_file(src_path, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_path = os.path.join(dest_folder, os.path.basename(src_path))
    if os.path.exists(dest_path):
        os.remove(dest_path)  # 删除已有文件
    shutil.move(src_path, dest_path)

# 获取网页内容
response = requests.get(base_url)
response.raise_for_status()
soup = BeautifulSoup(response.content, 'html.parser')

# 查找所有链接
links = soup.find_all('a')

# 下载并移动所有文件
for link in links:
    try:
        href = link.get('href')
        if href and href != '../':  # 排除上一级目录链接
            file_url = base_url + href
            print(f'Downloading {file_url}...')
            downloaded_file = download_file(file_url, download_dir)
            print(f'Moving {downloaded_file} to {target_dir}...')
            move_file(downloaded_file, target_dir)
    except:
        continue

print('All files downloaded and moved.')
