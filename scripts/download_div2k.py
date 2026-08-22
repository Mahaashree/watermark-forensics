"""
Download DIV2K validation set (mirror-first, fallback to official).
CLI: python scripts/download_div2k.py --output data/raw --max-images 800
"""

import argparse
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

MIRRORS = [
    "https://huggingface.co/datasets/eugenesiow/Div2k/resolve/main/DIV2K_valid_HR.zip",
    "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
]


def download_file(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-images", type=int, default=800)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    zip_path = args.output / "DIV2K_valid_HR.zip"

    for url in MIRRORS:
        print(f"Trying {url}...")
        if download_file(url, zip_path):
            break
    else:
        raise RuntimeError("All mirrors failed")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.output)

    extracted = list((args.output / "DIV2K_valid_HR").glob("*.png"))
    for i, src in enumerate(extracted[:args.max_images]):
        dst = args.output / f"{i:04d}.png"
        shutil.move(str(src), str(dst))

    shutil.rmtree(args.output / "DIV2K_valid_HR", ignore_errors=True)
    zip_path.unlink()

    print(f"Done: {len(list(args.output.glob('*.png')))} images in {args.output}")


if __name__ == "__main__":
    main()