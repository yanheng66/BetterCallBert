
import os
import time
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOVINFO_API_KEY")

BASE_URL = "https://api.govinfo.gov/packages"
SAVE_DIR = "data/govinfo"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_summary(title_num):
    package_id = f"USCODE-2022-title{title_num}"
    url = f"{BASE_URL}/{package_id}/summary"
    params = {"api_key": API_KEY}
    print(f"📦 Fetching summary for Title {title_num} ...")
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"❌ Title {title_num} not found.")
        return None
    return r.json()

def download_text(title_num, summary):
    downloads = summary.get("download", {})
    for ext in ["txtLink", "htmlLink", "xmlLink"]:
        if ext in downloads:
            txt_url = downloads[ext]
            break
    else:
        print(f"⚠️ No suitable download link for Title {title_num}")
        return

    print(f"⬇️ Downloading Title {title_num} from {txt_url}")
    r = requests.get(txt_url, params={"api_key": API_KEY})
    if r.status_code == 200:
        # 使用扩展名或 fallback 为 .txt
        suffix = os.path.splitext(urlparse(txt_url).path)[-1] or ".txt"
        save_path = os.path.join(SAVE_DIR, f"title-{str(title_num).zfill(2)}{suffix}")
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Saved Title {title_num} to {save_path}")
    else:
        print(f"❌ Failed to download Title {title_num}. Status: {r.status_code}")

def fetch_all_titles():
    for i in range(1, 55):
        summary = get_summary(i)
        if summary:
            download_text(i, summary)
        time.sleep(1)

if __name__ == "__main__":
    fetch_all_titles()
