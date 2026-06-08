"""Multithreaded image downloader using wget."""
import os
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd


def download_image(url: str, save_path: str, retries: int = 3) -> bool:
    """Download a single image with retries."""
    for _ in range(retries):
        try:
            result = subprocess.run(
                ["wget", "-q", "--timeout=10", "--tries=1", "-O", save_path, url],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if result.returncode == 0 and os.path.exists(save_path):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def download_all(df: pd.DataFrame, image_folder: str, max_workers: int = 16):
    """Download all images from df['image_link'] using thread pool."""
    os.makedirs(image_folder, exist_ok=True)
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for _, row in df.iterrows():
            url = row.get('image_link', '')
            if not isinstance(url, str) or not url.strip():
                continue

            ext = Path(url).suffix
            if not ext or len(ext) > 5:
                ext = ".jpg"

            save_path = os.path.join(image_folder, f"{row['sample_id']}{ext}")
            if os.path.exists(save_path):
                continue

            futures.append(pool.submit(download_image, url, save_path))

        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            f.result()

    print(f"✅ Done. Images saved in: {image_folder}")


if __name__ == "__main__":
    df = pd.read_csv('/home/user3/amazon/test_split_final.csv')
    download_all(df, "./test_images/", max_workers=32)
