import argparse
import json
import os
import socket
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from generate_answer import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL,
    build_user_prompt,
    format_evidence,
    generate_answer,
)
from search_hybrid import DEFAULT_CHUNKS_PATH, HybridRetriever


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NG222 RAG</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #f0f3f7;
      --text: #17202a;
      --muted: #5f6b7a;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --mono: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      --sans: Inter, Segoe UI, Arial, sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      line-height: 1.45;
    }

    header {
      border-bottom: 1px solid var(--line);
      background: var(--surface);
      padding: 14px 24px;
    }

    .header-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      max-width: 1320px;
      margin: 0 auto;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }

    .meta {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    main {
      max-width: 1320px;
      margin: 0 auto;
      padding: 20px 24px 32px;
      display: grid;
      grid-template-columns: minmax(340px, 0.95fr) minmax(420px, 1.45fr);
      gap: 18px;
    }

    section {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }

    .left, .right {
      padding: 16px;
    }

    label {
      display: block;
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 8px;
    }

    textarea {
      width: 100%;
      min-height: 136px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font: 14px var(--sans);
      color: var(--text);
      background: #fff;
    }

    textarea:focus, select:focus {
      outline: 2px solid rgba(15, 118, 110, 0.22);
      border-color: var(--accent);
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 14px;
    }

    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 9px 10px;
      font: 14px var(--sans);
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 16px;
    }

    button {
      min-height: 40px;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 9px 12px;
      font: 650 14px var(--sans);
      cursor: pointer;
    }

    button.primary {
      background: var(--accent);
      color: #fff;
    }

    button.primary:hover { background: var(--accent-dark); }

    button.secondary {
      background: var(--surface-2);
      color: var(--text);
      border-color: var(--line);
    }

    button.secondary:hover { background: #e6ebf2; }

    button:disabled {
      opacity: 0.62;
      cursor: not-allowed;
    }

    .status {
      margin-top: 12px;
      min-height: 22px;
      color: var(--muted);
      font-size: 13px;
    }

    .error {
      color: var(--danger);
      white-space: pre-wrap;
    }

    .answer {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      min-height: 160px;
      white-space: pre-wrap;
      font-size: 15px;
    }

    .evidence-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 18px 0 10px;
      gap: 12px;
    }

    h2 {
      font-size: 14px;
      margin: 0;
      font-weight: 700;
      letter-spacing: 0;
    }

    .evidence-list {
      display: grid;
      gap: 10px;
    }

    .evidence-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }

    .evidence-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 8px;
    }

    .chunk-id {
      font: 13px var(--mono);
      overflow-wrap: anywhere;
    }

    .score {
      color: var(--muted);
      font: 12px var(--mono);
      white-space: nowrap;
    }

    .section-path {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 7px;
    }

    .chunk-text {
      font-size: 13px;
      color: #253141;
    }

    @media (max-width: 900px) {
      main {
        grid-template-columns: 1fr;
        padding: 14px;
      }

      .header-row {
        align-items: flex-start;
        flex-direction: column;
      }

      .meta { white-space: normal; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-row">
      <h1>Clinical Guideline for Depression</h1>
      <div class="meta">Hybrid alpha 0.40 | top-k evidence</div>
    </div>
  </header>

  <main>
    <section class="left">
      <label for="query">Question</label>
      <textarea id="query">How should antidepressant medication be tapered when stopping?</textarea>

      <div class="controls">
        <div>
          <label for="topK">Top K</label>
          <select id="topK">
            <option value="3">3</option>
            <option value="5" selected>5</option>
            <option value="8">8</option>
          </select>
        </div>
        <div>
          <label for="model">Model</label>
          <select id="model">
            <option value="">env OPENAI_MODEL</option>
            <option value="gpt-4.1-mini">gpt-4.1-mini</option>
          </select>
        </div>
      </div>

      <div class="actions">
        <button id="retrieve" class="secondary" type="button">Retrieve</button>
        <button id="answer" class="primary" type="button">Answer</button>
      </div>

      <div id="status" class="status"></div>
    </section>

    <section class="right">
      <h2>Answer</h2>
      <div id="answerBox" class="answer"></div>

      <div class="evidence-head">
        <h2>Evidence</h2>
        <div id="evidenceCount" class="meta"></div>
      </div>
      <div id="evidenceList" class="evidence-list"></div>
    </section>
  </main>

  <script>
    const queryEl = document.getElementById("query");
    const topKEl = document.getElementById("topK");
    const modelEl = document.getElementById("model");
    const statusEl = document.getElementById("status");
    const answerBox = document.getElementById("answerBox");
    const evidenceList = document.getElementById("evidenceList");
    const evidenceCount = document.getElementById("evidenceCount");
    const retrieveBtn = document.getElementById("retrieve");
    const answerBtn = document.getElementById("answer");

    function setBusy(busy) {
      retrieveBtn.disabled = busy;
      answerBtn.disabled = busy;
      if (busy) {
        statusEl.textContent = "Running...";
        statusEl.className = "status";
      }
    }

    function showError(message) {
      statusEl.textContent = message;
      statusEl.className = "status error";
    }

    function renderEvidence(items) {
      evidenceList.innerHTML = "";
      evidenceCount.textContent = items.length ? `${items.length} chunks` : "";
      for (const item of items) {
        const row = document.createElement("div");
        row.className = "evidence-item";
        row.innerHTML = `
          <div class="evidence-top">
            <div class="chunk-id">${escapeHtml(item.chunk_id)}</div>
            <div class="score">${Number(item.score).toFixed(4)}</div>
          </div>
          <div class="section-path">${escapeHtml(item.section_path || "")}</div>
          <div class="chunk-text">${escapeHtml(item.text || "")}</div>
        `;
        evidenceList.appendChild(row);
      }
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function run(withAnswer) {
      const query = queryEl.value.trim();
      if (!query) {
        showError("Question is required.");
        return;
      }

      setBusy(true);
      if (!withAnswer) answerBox.textContent = "";

      try {
        const response = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            top_k: Number(topKEl.value),
            generate_answer: withAnswer,
            model: modelEl.value || null
          })
        });

        const data = await response.json();
        if (!response.ok) {
          renderEvidence(data.evidence || []);
          answerBox.textContent = "";
          throw new Error(data.error || `HTTP ${response.status}`);
        }

        renderEvidence(data.evidence || []);
        answerBox.textContent = data.answer || "";
        statusEl.textContent = withAnswer ? "Answer generated." : "Evidence retrieved.";
        statusEl.className = "status";
      } catch (error) {
        showError(error.message);
      } finally {
        retrieveBtn.disabled = false;
        answerBtn.disabled = false;
      }
    }

    retrieveBtn.addEventListener("click", () => run(false));
    answerBtn.addEventListener("click", () => run(true));
  </script>
</body>
</html>
"""


class RagUiServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, retriever, model):
        super().__init__(server_address, handler_class)
        self.retriever = retriever
        self.model = model


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        if self.path in {"/", "/index.html"}:
            self.send_bytes(HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        self.send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        if self.path != "/api/query":
            self.send_json({"error": "Not found"}, status=404)
            return

        try:
            payload = self.read_json()
            query = str(payload.get("query", "")).strip()
            if not query:
                self.send_json({"error": "Question is required"}, status=400)
                return

            top_k = int(payload.get("top_k", 5))
            top_k = max(1, min(top_k, 10))
            should_answer = bool(payload.get("generate_answer", False))
            model = payload.get("model") or self.server.model

            results = self.server.retriever.search(query, top_k=top_k)
            evidence_items = [self.evidence_item(score, chunk) for score, chunk in results]

            answer = ""
            if should_answer:
                evidence = format_evidence(results, max_chars=1800)
                prompt = build_user_prompt(query, evidence)
                answer = generate_answer(model, prompt)

            self.send_json(
                {
                    "query": query,
                    "model": model,
                    "answer": answer,
                    "evidence": evidence_items,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            evidence = locals().get("evidence_items", [])
            self.send_json({"error": str(exc), "evidence": evidence}, status=500)

    def evidence_item(self, score, chunk):
        return {
            "score": score,
            "chunk_id": chunk.get("chunk_id", ""),
            "source_id": chunk.get("source_id", ""),
            "section_path": chunk.get("section_path", ""),
            "anchor": chunk.get("anchor", ""),
            "text": chunk.get("text", ""),
        }

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def send_json(self, payload, status=200):
        self.send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def send_bytes(self, body, status=200, content_type="application/octet-stream"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def find_port(host, preferred_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex((host, preferred_port)) != 0:
            return preferred_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a local web UI for NG222 hybrid RAG.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    return parser.parse_args()


def main():
    args = parse_args()
    port = find_port(args.host, args.port)
    retriever = HybridRetriever(
        chunks_path=args.chunks,
        index_dir=args.index_dir,
        alpha=args.alpha,
        candidate_k=args.candidate_k,
    )
    server = RagUiServer((args.host, port), Handler, retriever=retriever, model=args.model)
    url = f"http://{args.host}:{port}"
    print(f"NG222 RAG UI running at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
