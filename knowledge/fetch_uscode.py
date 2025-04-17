import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOVINFO_API_KEY")

YEAR = "2022"
SAVE_DIR = "data/uscode"


def get_package_summary(title_num):
    package_id = f"USCODE-{YEAR}-title{title_num}"
    url = f"https://api.govinfo.gov/packages/{package_id}/summary"
    params = {"api_key": API_KEY}

    print(f"📡 Requesting {package_id} ...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f" Title {title_num} not found. Status: {response.status_code}")
        print(f" Full request URL: {response.url}")
        print(" Response text:", response.text)
        return None

    return response.json()


def download_pdf_from_summary(title_num, summary):
    pdf_url = summary.get("download", {}).get("pdfLink", None)
    if not pdf_url:
        print(f"No PDF for Title {title_num}")
        return

    response = requests.get(pdf_url, params={"api_key": API_KEY})

    if response.status_code == 200:
        filename = os.path.join(SAVE_DIR, f"uscode-title{title_num}.pdf")
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Title {title_num} downloaded.")
    else:
        print(f"❌ Failed to download Title {title_num} PDF. Status: {response.status_code}")
        print("Response text:", response.text)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    for title_num in range(1, 55):
        print(f"📦 Processing Title {title_num}...")
        summary = get_package_summary(title_num)
        if summary:
            download_pdf_from_summary(title_num, summary)
        time.sleep(1)

if __name__ == "__main__":
    main()
