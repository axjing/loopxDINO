"""
download_voc.py
Downloads and extracts Pascal VOC 2012 dataset into data/pascal_voc/
Supports HTTP range resumable download and tar extraction.
"""

import os
import tarfile
import time
import urllib.request

VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
DEST_DIR = os.path.abspath("data/pascal_voc")
TAR_PATH = os.path.join(DEST_DIR, "VOCtrainval_11-May-2012.tar")


def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    existing_bytes = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    req = urllib.request.Request(url)
    if existing_bytes > 0:
        req.add_header("Range", f"bytes={existing_bytes}-")
        print(f"[VOC Download] Resuming download from {existing_bytes / 1e6:.1f} MB...")

    try:
        response = urllib.request.urlopen(req, timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"[VOC Download] Range request failed ({e}), starting fresh download...")
        req = urllib.request.Request(url)
        existing_bytes = 0
        response = urllib.request.urlopen(req, timeout=30)

    content_range = response.headers.get("Content-Range")
    if content_range:
        total_size = int(content_range.split("/")[-1])
    else:
        content_length = response.headers.get("Content-Length")
        total_size = int(content_length) + existing_bytes if content_length else 0

    print(f"[VOC Download] Target size: {total_size / 1e6:.1f} MB")

    mode = "ab" if existing_bytes > 0 else "wb"
    downloaded = existing_bytes
    chunk_size = 1024 * 1024  # 1 MB chunk
    last_log = time.time()

    with open(dest_path, mode) as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log >= 15:  # log every 15s
                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                print(
                    f"[VOC Download] Progress: {downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB ({pct:.1f}%)"
                )
                last_log = now

    print(f"[VOC Download] Finished downloading: {downloaded / 1e6:.1f} MB")


def extract_tar(tar_path: str, extract_dir: str):
    print(f"[VOC Extract] Extracting {tar_path} into {extract_dir}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)
    print("[VOC Extract] Extraction complete! VOCdevkit is ready.")


def main():
    expected_devkit = os.path.join(DEST_DIR, "VOCdevkit", "VOC2012")
    if os.path.exists(expected_devkit):
        print(f"[VOC Ready] VOCdevkit already exists at {expected_devkit}!")
        return

    print("==================================================")
    print(" Pascal VOC 2012 Dataset Setup")
    print(f" Target Directory: {DEST_DIR}")
    print("==================================================")

    download_file(VOC_URL, TAR_PATH)
    extract_tar(TAR_PATH, DEST_DIR)
    print("[VOC Ready] Dataset setup successfully finished!")


if __name__ == "__main__":
    main()
