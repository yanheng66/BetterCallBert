import os
import time
import requests
import re
import json
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOVINFO_API_KEY")

# Define paths
BASE_URL = "https://api.govinfo.gov/packages"
SAVE_DIR = "data/govinfo"
OUTPUT_DIR = "data/parsed_sections_v2"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper function to clean text
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

# Function to fetch summary of each title
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

# Function to download the text file for a title
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
        suffix = os.path.splitext(urlparse(txt_url).path)[-1] or ".txt"
        save_path = os.path.join(SAVE_DIR, f"title-{str(title_num).zfill(2)}{suffix}")
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Saved Title {title_num} to {save_path}")
    else:
        print(f"❌ Failed to download Title {title_num}. Status: {r.status_code}")

# Function to parse one HTML file and extract sections
def parse_one_html(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    title_match = re.search(r"title-(\d+)", filepath.name)
    title_id = title_match.group(1).zfill(2) if title_match else "00"
    section_count = 0

    all_sections = soup.find_all("h3", class_="section-head")
    for section_head in all_sections:
        section_title = clean_text(section_head.get_text())
        section_number = re.findall(r"§\s*(\d+[A-Za-z\d\-]*)", section_title)
        section_id = section_number[0] if section_number else f"unknown-{section_count}"

        content = []
        node = section_head.find_next_sibling()
        while node and node.name != "h3":
            if node.name == "p" and any(cls in node.get("class", []) for cls in [
                "statutory-body", "statutory-body-1em", "source-credit", "note-body"
            ]):
                content.append(clean_text(node.get_text()))
            node = node.find_next_sibling()

        if content:
            data = {
                "title": f"Title {int(title_id)}",
                "section": f"§{section_id}",
                "heading": section_title,
                "text": "\n".join(content)
            }
            save_path = os.path.join(OUTPUT_DIR, f"title-{title_id}-section-{section_id}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            section_count += 1

    return section_count

# Function to fetch and parse all titles
def fetch_and_parse_all_titles():
    for i in range(1, 55):
        summary = get_summary(i)
        if summary:
            download_text(i, summary)
            # Parse the downloaded HTML text file
            html_file_path = os.path.join(SAVE_DIR, f"title-{str(i).zfill(2)}.html")
            count = parse_one_html(html_file_path)
            print(f"✅ Parsed {count} sections from Title {i}")
        time.sleep(1)

if __name__ == "__main__":
    fetch_and_parse_all_titles()
