"""
Section-aware PDF -> RAG chunker (tuned for ISLP, works for most LaTeX books).
Usage: python chunk_islp.py input.pdf chunks.jsonl [--max-tokens 500 --overlap 60]
"""
import argparse, json, re
import pymupdf, tiktoken

try:
    ENC = tiktoken.get_encoding("cl100k_base")
    ntok = lambda s: len(ENC.encode(s))
except Exception:  # offline: approximate (~4 chars/token for English prose)
    print("warning: tiktoken vocab unavailable, using len/4 estimate")
    ntok = lambda s: max(1, len(s) // 4)
SKIP_SECTIONS = {"Contents", "Index"}

def page_lines(page):
    """Body text lines only: drop running headers, margin notes, figure internals."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = l["spans"]
            if not spans:
                continue
            x0, y0, x1, y1 = l["bbox"]
            font, size = spans[0]["font"], spans[0]["size"]
            # margin notes can share a line with body text -> filter per span
            spans = [s for s in spans if not (s["bbox"][0] >= 418 and s["size"] < 9)]
            if not spans:
                continue
            font, size = spans[0]["font"], spans[0]["size"]
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            if y1 < 50:                                  # running header / page number
                continue
            if x0 >= 418 and size < 9:                   # margin keyword notes
                continue
            if "Arial" in font or "Helvetica" in font:   # axis labels inside figures
                continue
            is_code = "MonoLt" in font or ("Mono" in font and size < 8.6)
            out.append(("\x00" + text) if is_code else text)  # \x00 marks a code line
    return out

def clean(text):
    out, code, prose = [], [], []
    def flush_prose():
        if prose: out.append(_clean_prose("\n".join(prose))); prose.clear()
    def flush_code():
        if code: out.append("```python\n" + "\n".join(code) + "\n```"); code.clear()
    for ln in text.split("\n"):
        if ln.startswith("\x00"):
            flush_prose(); code.append(ln[1:])
        else:
            flush_code(); prose.append(ln)
    flush_prose(); flush_code()
    return "\n".join(o for o in out if o)

def _clean_prose(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)         # de-hyphenate line breaks
    text = re.sub(r"(?<![.:?!])\n(?!\n)", " ", text)     # rejoin wrapped lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def build_sections(doc):
    toc = doc.get_toc()  # [level, title, page(1-based)]
    sections = []
    for i, (lvl, title, start) in enumerate(toc):
        end = next((p for l2, _, p in toc[i + 1:] if True), doc.page_count + 1)
        # path = chapter > section > subsection
        path, cur = [], lvl
        for l2, t2, _ in reversed(toc[: i + 1]):
            if l2 <= cur:
                path.insert(0, t2); cur = l2 - 1
            if cur == 0:
                break
        sections.append({"title": title, "path": path, "start": start, "end": end,
                         "next_title": toc[i + 1][1] if i + 1 < len(toc) else None})
    return sections

def section_text(doc, sec, cache):
    """Text from this heading up to the next heading (headings can start mid-page)."""
    pages = range(sec["start"], max(sec["start"], sec["end"]) + 1)
    parts = []
    for p in pages:
        if p > doc.page_count:
            break
        if p not in cache:
            cache[p] = page_lines(doc[p - 1])
        parts.append((p, cache[p]))
    # flatten with page tags
    flat = [(p, ln) for p, lines in parts for ln in lines]
    norm = lambda s: re.sub(r"\W+", "", s.replace("\x00", "")).lower()
    def find(title, frm):
        t = norm(title)
        for i in range(frm, len(flat)):
            for k in (1, 2, 3):   # heading may be split: "3.6.2" / "Simple Linear Regression"
                joined = norm("".join(ln for _, ln in flat[i:i + k]))
                if joined == t:
                    return i, k
        return None, 0
    si, sk = find(sec["title"], 0)
    start = (si + sk) if si is not None else 0
    end = len(flat)
    if sec["next_title"]:
        ni, _ = find(sec["next_title"], start)
        if ni is not None:
            end = ni
    return flat[start:end]

def chunk_section(flat, max_tokens, overlap):
    # group lines into paragraphs-ish units, keep page provenance
    units, buf, pg = [], [], None
    for p, ln in flat:
        buf.append(ln); pg = pg or p
        if re.search(r"[.:?!]$", ln) and ntok(" ".join(buf)) > 40:
            units.append((pg, p, clean("\n".join(buf)))); buf, pg = [], None
    if buf:
        units.append((pg, flat[-1][0], clean("\n".join(buf))))
    chunks, cur, cur_tok = [], [], 0
    for u in units:
        t = ntok(u[2])
        if cur and cur_tok + t > max_tokens:
            chunks.append(cur)
            # carry tail units as overlap
            tail, tt = [], 0
            for x in reversed(cur):
                tt += ntok(x[2])
                if tt > overlap: break
                tail.insert(0, x)
            cur, cur_tok = tail, sum(ntok(x[2]) for x in tail)
        cur.append(u); cur_tok += t
    if cur:
        chunks.append(cur)
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("out")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=60)
    a = ap.parse_args()

    doc = pymupdf.open(a.pdf)
    secs, cache, n = build_sections(doc), {}, 0
    with open(a.out, "w") as f:
        for sec in secs:
            if sec["path"][0] in SKIP_SECTIONS:
                continue
            flat = section_text(doc, sec, cache)
            if not flat:
                continue
            for ch in chunk_section(flat, a.max_tokens, a.overlap):
                text = " ".join(u[2] for u in ch)
                if ntok(text) < 30:
                    continue
                breadcrumb = " > ".join(sec["path"])
                rec = {
                    "id": f"islp-{n:05d}",
                    "text": text,
                    "embed_text": f"{breadcrumb}\n\n{text}",   # heading context helps retrieval
                    "section": breadcrumb,
                    "page_start": ch[0][0], "page_end": ch[-1][1],
                    "is_lab": "Lab:" in breadcrumb,
                    "tokens": ntok(text),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} chunks -> {a.out}")

if __name__ == "__main__":
    main()
