import os
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv


META_DATASET = 'Sports_and_Outdoors'


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


asin2item_path = os.path.join(PROJECT_ROOT, 'dict', META_DATASET, 'asin2itemID.csv')
item_final_path = os.path.join(PROJECT_ROOT, 'dict', META_DATASET, 'item_final.csv')
asinlist_path = os.path.join(PROJECT_ROOT, 'datasets', META_DATASET, 'asinlist.npy')

asin2item_df = pd.read_csv(asin2item_path)
item_final_df = pd.read_csv(item_final_path)

asin_list = np.load(asinlist_path, allow_pickle=True)

filtered_item_final_df = item_final_df[item_final_df['asin'].isin(asin_list)]

asin_to_url = pd.Series(filtered_item_final_df.image.values, index=filtered_item_final_df.asin).to_dict()

# # 只重新下载之前失败的这几个 ASIN（如果想重新下全部，注释掉下面三行即可）
# retry_asins = ['B000Q1QCBE', 'B000RDHF5S', 'B00LM9DA6E', 'B014HG74GK']
# asin_to_url = {asin: url for asin, url in asin_to_url.items() if asin in retry_asins}
# print("只重新下载这些 ASIN：", list(asin_to_url.keys()))


image_path = os.path.join(PROJECT_ROOT, 'datasets', META_DATASET, 'images')
if not os.path.exists(image_path):
    os.makedirs(image_path)

print(f"Total ASINs in asin_list: {len(asin_list)}")
print(f"Total ASINs in filtered_item_final_df: {len(filtered_item_final_df)}")

if len(asin_to_url) != len(filtered_item_final_df):
    print("Warning: There are duplicate ASINs in the dataset.")

failed_asins = []

def download_image(asin, url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(image_path, f"{asin}.jpg")
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
        else:
            print(f"Failed to download image for ASIN: {asin}, URL: {url}")
            return asin
    except Exception as e:
        print(f"Error downloading image for ASIN: {asin}, URL: {url}, Error: {e}")
        return asin
    return None

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_image, asin, url): asin for asin, url in asin_to_url.items()}

    for future in tqdm(as_completed(futures), total=len(futures)):
        asin = futures[future]
        if future.result():
            failed_asins.append(asin)

print (failed_asins)
with open("failed_asins.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Failed ASINs"])  
    for asin in failed_asins:
        writer.writerow([asin])

print("Failed ASINs have been written to 'failed_asins.csv'.")



downloaded_images = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
print(f"Total images downloaded successfully: {downloaded_images}")
print(f"Total failed downloads: {len(failed_asins)}")

