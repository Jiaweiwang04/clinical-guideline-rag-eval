import json
import os
from bs4 import BeautifulSoup

REGISTRY_META = "data/raw/ng222/metadata.json"
OUT_PATH = "data/processed/ng222_chunks.jsonl"


def load_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_html(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def norm_space(s: str) -> str:
    return " ".join(s.split())


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    meta = load_metadata(REGISTRY_META)
    source_id = meta["source_id"]
    html_path = meta["local_files"]["recommendations_html"]

    html = read_html(html_path)
    soup = BeautifulSoup(html, "lxml")

    

    chunks = []

    # Strategy:
    # - Track current section heading from h2/h3/h4
    # - Extract recommendation-like blocks with stable ids/anchors
    # NICE pages usually have anchors for recommendations; common patterns:
    #  - elements with id containing 'recommendation'
    #  - or <a id="recommendation-..."> / headings with that id
    # We'll:
    # 1) collect all elements that have an id containing 'recommendation'
    # 2) for each, take nearby text content (itself + following siblings until next anchor/heading)

    # Pre-collect headings to infer section path
    headings = soup.find_all(["h2", "h3", "h4"])
    heading_positions = []
    for h in headings:
        heading_positions.append(h)

    def get_current_section(el):
        # Find nearest preceding heading (h2/h3/h4)
        prev = el.find_previous(["h2", "h3", "h4"])
        if prev:
            return norm_space(prev.get_text(" ", strip=True))
        return "Recommendations"

    # get ng222 anchors globally (prefer numbered ones) ---
    all_with_id = soup.find_all(attrs={"id": True})
    ng_anchors = []
    for el in all_with_id:
        rid = (el.get("id") or "").strip()
        if rid.lower().startswith("ng222-"):
            ng_anchors.append(el)

    def is_numbered_anchor(rid: str) -> bool:
        # example: ng222-1_2_12
        r = rid.lower()
        if not r.startswith("ng222-"):
            return False
        tail = r[len("ng222-"):]
        return any(ch.isdigit() for ch in tail) and "_" in tail

    # split anchors: numbered (recommendation items) vs section-like
    seen = set()
    numbered = []
    section_like = []
    for a in ng_anchors:
        rid = (a.get("id") or "").strip()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        if rid.lower() == "ng222-recommendations":
            continue
        if is_numbered_anchor(rid):
            numbered.append(a)
        else:
            section_like.append(a)

    print(f"[DEBUG] ng222 anchors total: {len(ng_anchors)}")
    print(f"[DEBUG] numbered anchors: {len(numbered)} | section-like anchors: {len(section_like)}")

    rec_anchors = numbered  # use numbered anchors as chunk starts
    seen = set()
    for a in rec_anchors:
        rid = a.get("id", "").strip()
        if not rid or rid in seen:
            continue
        seen.add(rid)

        section = get_current_section(a)
        anchor = f"#{rid}"

        # Gather text: anchor element + a limited window of following content
        # Stop when we hit the next recommendation anchor or a new heading of same/higher level.
        texts = []

        # include self text if meaningful
        self_text = a.get_text(" ", strip=True)
        if self_text:
            texts.append(self_text)

        # walk following siblings (or next elements) to collect recommendation body
        # We cap by number of characters to avoid huge chunks.
        char_cap = 1800
        total = len(" ".join(texts))

        cur = a
        while total < char_cap:
            nxt = cur.find_next()
            if nxt is None:
                break

            # stop if next is another recommendation anchor
            nid = nxt.get("id", "")
            if isinstance(nid, str) and nid and "recommendation" in nid.lower() and nid != rid:
                break

            # stop at next heading (new section)
            if nxt.name in ["h2", "h3", "h4"]:
                break

            # collect paragraph / list items mainly
            if nxt.name in ["p", "li"]:
                t = norm_space(nxt.get_text(" ", strip=True))
                if t:
                    texts.append(t)
                    total += len(t) + 1

            cur = nxt

        text = norm_space("\n".join(texts))
        if len(text) < 30:
            continue

        chunk_id = f"{source_id}|{rid}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "source_id": source_id,
                "doc": "recommendations_html",
                "section_path": section,
                "anchor": anchor,
                "text": text,
            }
        )

    # Write JSONL
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {len(chunks)} chunks -> {OUT_PATH}")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- chunk {i+1} ---")
        print("section:", c["section_path"])
        print("anchor :", c["anchor"])
        print("text   :", c["text"][:160], "...")


if __name__ == "__main__":
    main()