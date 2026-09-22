"""Fallback ingestion for environments where linked PDFs are unavailable.
It indexes the annual FCA table rows only. Use ingest.py for the full assignment."""
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.rag.chunker import chunk_text
from app.rag.store import VectorStore
chunks=[]
for p in sorted((ROOT/'data'/'raw').glob('*_table.json')):
    year=int(p.stem.split('_')[0])
    for row in json.loads(p.read_text()):
        text=f"Firm: {row['firm']}\nDate: {row['date']}\nAmount: {row['amount_text']}\nReason: {row['reason']}"
        chunks.extend(chunk_text(text,{**row,'title':f"FCA {year} Fines"}))
VectorStore(str(ROOT/'data'/'processed')).build(chunks)
print('Indexed',len(chunks),'summary chunks')
