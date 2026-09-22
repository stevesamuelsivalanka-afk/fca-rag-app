"""Download FCA annual fines pages and linked Final Notice PDFs, extract text, chunk, embed and build FAISS."""
from pathlib import Path
import json, re, sys, requests, fitz
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.rag.chunker import chunk_text
from app.rag.store import VectorStore

YEARS = [2024, 2025, 2026]
BASE = "https://www.fca.org.uk/news/news-stories/{year}-fines"
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def money(value):
    m = re.search(r"£([0-9][0-9,]*(?:\.[0-9]+)?)", value or "")
    return float(m.group(1).replace(",", "")) if m else None


def normalize_firm(name):
    return re.sub(r"\s+", " ", name.lower()).strip()


def scrape_year(year):
    html = requests.get(BASE.format(year=year), timeout=60, headers={"User-Agent":"fca-rag-takehome/1.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    table = soup.find("table")
    if not table:
        raise RuntimeError(f"No fines table found for {year}")
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th","td"])
        if len(cells) < 4:
            continue
        link = cells[0].find("a")
        firm = cells[0].get_text(" ", strip=True)
        if not link:
            continue
        rows.append({"firm": firm, "firm_normalized": normalize_firm(firm), "date": cells[1].get_text(" ", strip=True), "amount_text": cells[2].get_text(" ", strip=True), "amount": money(cells[2].get_text(" ", strip=True)), "reason": cells[3].get_text(" ", strip=True), "url": requests.compat.urljoin(BASE.format(year=year), link.get("href")), "year": year})
    return rows


def download_pdf(url, path):
    r = requests.get(url, timeout=90, headers={"User-Agent":"fca-rag-takehome/1.0"})
    r.raise_for_status()
    path.write_bytes(r.content)


def main():
    RAW.mkdir(parents=True, exist_ok=True); PROC.mkdir(parents=True, exist_ok=True)
    all_chunks=[]; records=[]
    for year in YEARS:
        rows = scrape_year(year)
        (RAW / f"{year}_table.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),encoding="utf-8")
        for i, row in enumerate(rows):
            pdf_dir = RAW / str(year); pdf_dir.mkdir(exist_ok=True)
            pdf_path = pdf_dir / f"{i:03d}.pdf"
            try:
                download_pdf(row["url"], pdf_path)
                doc = fitz.open(pdf_path)
                for page_no, page in enumerate(doc, 1):
                    text = page.get_text("text")
                    meta = {**row, "title": f"FCA Final Notice: {row['firm']}", "page": page_no, "pdf_path": str(pdf_path.relative_to(ROOT))}
                    all_chunks.extend(chunk_text(text, meta))
            except Exception as exc:
                # Keep the annual table data even if an individual PDF fails.
                print(f"WARNING: {year} {row['firm']}: {exc}")
    store = VectorStore(str(PROC))
    store.build(all_chunks)
    print(f"Indexed {len(all_chunks)} chunks")

if __name__ == "__main__":
    main()
