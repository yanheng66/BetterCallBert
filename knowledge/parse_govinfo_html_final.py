
import os
import re
import json
from bs4 import BeautifulSoup
from pathlib import Path

INPUT_DIR = "data/govinfo"
OUTPUT_DIR = "data/parsed_sections_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

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

def parse_all_htmls():
    files = sorted([
        f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")
    ], key=lambda x: int(re.findall(r"title-(\d+)", x)[0]))

    total = 0
    for file in files:
        count = parse_one_html(Path(INPUT_DIR) / file)
        print(f"✅ Parsed {count} sections from {file}")
        total += count
    print(f"📚 Total parsed sections: {total}")

if __name__ == "__main__":
    parse_all_htmls()
