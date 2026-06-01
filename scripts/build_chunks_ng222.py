import json
import os
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup


METADATA_PATH = Path("data/raw/ng222/metadata.json")
OUT_PATH = Path("data/processed/ng222_chunks.jsonl")


TEXT_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u2022": "-",
    "\u00a0": " ",
    "鈥?": "-",
    "â€¢": "-",
}


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_html(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    for old, new in TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_recommendation_id(value: str) -> bool:
    return bool(re.fullmatch(r"ng222-\d+_\d+_\d+", value or ""))


def section_path_for(article) -> str:
    section = article.find_parent("div", class_="section")
    while section:
        title = section.get("title")
        if title and not title.lower().startswith("table "):
            return clean_text(title)
        section = section.find_parent("div", class_="section")
    heading = article.find_previous(["h2", "h3", "h4"])
    if heading:
        return clean_text(heading.get_text(" ", strip=True))
    return "Recommendations"


def article_text(article) -> str:
    number_el = article.find(class_="recommendation__number")
    body_el = article.find(class_="recommendation__body")

    number = clean_text(number_el.get_text(" ", strip=True)) if number_el else ""
    body = clean_text(body_el.get_text(" ", strip=True)) if body_el else ""

    if number and body.startswith(number):
        return body
    return clean_text(f"{number} {body}")


def build_recommendation_chunks() -> list[dict]:
    meta = load_metadata(METADATA_PATH)
    html_path = Path(meta["local_files"]["recommendations_html"])
    soup = BeautifulSoup(read_html(html_path), "lxml")
    source_id = meta["source_id"]

    chunks = []
    seen = set()
    for article in soup.find_all("article", id=True):
        rid = article.get("id", "").strip()
        if not is_recommendation_id(rid) or rid in seen:
            continue
        seen.add(rid)

        text = article_text(article)
        if len(text) < 30:
            continue

        chunks.append(
            {
                "chunk_id": f"{source_id}|{rid}",
                "source_id": source_id,
                "doc": "recommendations_html",
                "section_path": section_path_for(article),
                "anchor": f"#{rid}",
                "text": text,
            }
        )

    return chunks


def write_jsonl(chunks: list[dict], out_path: Path) -> None:
    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    chunks = build_recommendation_chunks()
    write_jsonl(chunks, OUT_PATH)

    print(f"[OK] Wrote {len(chunks)} chunks -> {OUT_PATH}")
    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- chunk {i} ---")
        print("section:", chunk["section_path"])
        print("anchor :", chunk["anchor"])
        print("text   :", chunk["text"][:180], "...")


if __name__ == "__main__":
    main()
