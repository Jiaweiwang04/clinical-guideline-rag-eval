import json
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup

from build_chunks_ng222 import (
    METADATA_PATH,
    build_recommendation_chunks,
    clean_text,
    load_metadata,
    read_html,
)


OUT_PATH = Path("data/processed/ng222_chunks_with_tables.jsonl")


TABLE_META = {
    "table-1": {
        "table_id": "table_1",
        "section_path": "Table 1",
        "severity": "Less severe depression",
    },
    "table-2": {
        "table_id": "table_2",
        "section_path": "Table 2",
        "severity": "More severe depression",
    },
}


def slugify(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def extract_cell_text(cell) -> str:
    return clean_text(cell.get_text(" ", strip=True))


def label_text(text: str) -> str:
    return clean_text(text).rstrip("?:.")


def sentence(text: str) -> str:
    text = clean_text(text).rstrip(".")
    return f"{text}."


def table_headers(table) -> list[str]:
    header_row = table.find("thead").find("tr")
    return [label_text(th.get_text(" ", strip=True)) for th in header_row.find_all("th")]


def build_table_chunks() -> list[dict]:
    meta = load_metadata(METADATA_PATH)
    soup = BeautifulSoup(read_html(Path(meta["local_files"]["recommendations_html"])), "lxml")
    source_id = meta["source_id"]

    chunks = []
    for anchor_id, table_meta in TABLE_META.items():
        heading = soup.find(id=anchor_id)
        if not heading:
            continue

        section = heading.find_parent("div", class_="section")
        table = section.find("table") if section else None
        if not table:
            continue

        headers = table_headers(table)
        body = table.find("tbody") or table
        rows = body.find_all("tr", recursive=False)

        row_number = 0
        for row in rows:
            cells = row.find_all(["td", "th"], recursive=False)
            if len(cells) < len(headers):
                continue

            values = [extract_cell_text(cell) for cell in cells[: len(headers)]]
            if not values or not values[0]:
                continue

            row_number += 1
            row_label = values[0]
            parts = [
                f"{table_meta['section_path']}.",
                f"{table_meta['severity']}.",
                sentence(f"{headers[0]}: {row_label}"),
            ]
            for header, value in zip(headers[1:], values[1:]):
                if value:
                    parts.append(sentence(f"{header}: {value}"))

            table_id = table_meta["table_id"]
            anchor = f"#{anchor_id}-{slugify(row_label)}"
            chunks.append(
                {
                    "chunk_id": f"{source_id}|{table_id}|{row_number}",
                    "source_id": source_id,
                    "doc": "recommendations_html_table",
                    "section_path": table_meta["section_path"],
                    "table_id": table_id,
                    "row_label": row_label,
                    "anchor": anchor,
                    "text": clean_text(" ".join(parts)),
                }
            )

    return chunks


def write_jsonl(chunks: list[dict], out_path: Path) -> None:
    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    recommendation_chunks = build_recommendation_chunks()
    table_chunks = build_table_chunks()
    chunks = recommendation_chunks + table_chunks
    write_jsonl(chunks, OUT_PATH)

    print(
        f"[OK] Wrote {len(chunks)} chunks "
        f"({len(recommendation_chunks)} recommendations, {len(table_chunks)} table rows) "
        f"-> {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
