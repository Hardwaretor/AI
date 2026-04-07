#!/usr/bin/env python3
"""
Indexador RAG para la documentación de three.js

Descarga páginas bajo https://threejs.org/docs/, extrae texto, lo divide en fragmentos,
genera embeddings con `sentence-transformers` y crea un índice FAISS junto a un archivo
de metadatos (JSON) para búsqueda semántica local.

Uso:
  python index_docs.py --output-dir ./data --max-pages 300

Requisitos: ver `requirements.txt` en este mismo directorio.
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("ERROR: sentence_transformers no está instalado.")
    raise

try:
    import faiss
except Exception:
    faiss = None

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def fetch_links(start_url: str, max_pages: int = 300) -> list[str]:
    to_visit = [start_url]
    visited = set()
    collected = []
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"skip {url}: {e}")
            visited.add(url)
            continue
        visited.add(url)
        collected.append(url)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # keep only links under /docs
            if href.startswith("/docs") or ("threejs.org/docs" in href):
                if href.startswith("/"):
                    full = urljoin("https://threejs.org", href)
                else:
                    full = href
                if full not in visited and full not in to_visit:
                    to_visit.append(full)
    return collected


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.find("div", {"id": "content"}) or soup
    parts = []
    for tag in main.find_all(["h1", "h2", "h3", "p", "li", "pre", "code"]):
        t = tag.get_text(separator=" ", strip=True)
        if t:
            parts.append(t)
    return "\n".join(parts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_index(texts: list[str], model_name: str, output_dir: str, no_faiss: bool = False, batch_size: int = 32) -> tuple[object | None, np.ndarray]:
    model = SentenceTransformer(model_name)
    # encode in batches to reduce peak memory
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embs.append(emb)
    embeddings = np.vstack(all_embs)
    # normalize for cosine similarity with inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    if no_faiss:
        return None, embeddings

    if faiss is None:
        raise RuntimeError("faiss no está disponible; instale faiss-cpu o faiss-gpu, o use --no-faiss")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def main():
    parser = argparse.ArgumentParser(description="Index three.js docs for RAG (FAISS)")
    parser.add_argument("--start-url", default="https://threejs.org/docs/", help="URL inicial de docs")
    parser.add_argument("--output-dir", default="./data", help="Directorio de salida para índice y metadatos")
    parser.add_argument("--max-pages", type=int, default=300, help="Máximo número de páginas a rastrear")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Tamaño en caracteres por fragmento")
    parser.add_argument("--overlap", type=int, default=200, help="Solapamiento en caracteres entre fragmentos")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo de embeddings para sentence-transformers")
    parser.add_argument("--no-faiss", action="store_true", help="No construir índice FAISS (guarda solo embeddings + metadatos)")
    parser.add_argument("--embed-batch", type=int, default=32, help="Tamaño de lote para generar embeddings")
    parser.add_argument("--download-only", action="store_true", help="Solo descargar y guardar textos crudos sin generar embeddings")
    parser.add_argument("--delay", type=float, default=0.5, help="Segundos de retardo entre peticiones HTTP para reducir carga (default 0.5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Recolectando URLs desde:", args.start_url)
    urls = fetch_links(args.start_url, max_pages=args.max_pages)
    print(f"URLs recolectadas: {len(urls)}")

    docs = []
    for url in tqdm(urls, desc="Descargando"):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"fallo {url}: {e}")
            continue
        # throttle requests to avoid overloading the machine/network
        try:
            time.sleep(args.delay)
        except Exception:
            pass
        text = extract_text(r.text)
        if not text:
            continue
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        for i, chunk in enumerate(chunks):
            docs.append({"url": url, "chunk_index": i, "text": chunk})

    print(f"Fragmentos totales: {len(docs)}")
    raw_path = os.path.join(args.output_dir, "threejs_docs_raw.json")
    # If user requested download-only, save raw fragments and exit early to avoid embeddings
    if args.download_only:
        print("--download-only activado: guardando textos crudos en:", raw_path)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        print("Descarga completada. No se generaron embeddings ni índice.")
        return
    if not docs:
        print("No hay fragmentos para indexar. Fin.")
        return

    texts = [d["text"] for d in docs]
    print("Generando embeddings con:", args.model)
    index, embeddings = build_index(texts, args.model, args.output_dir, no_faiss=args.no_faiss, batch_size=args.embed_batch)

    idx_path = os.path.join(args.output_dir, "threejs_faiss.index")
    meta_path = os.path.join(args.output_dir, "threejs_docs_meta.json")
    npy_path = os.path.join(args.output_dir, "threejs_embeddings.npy")

    if index is not None:
        print("Guardando índice FAISS en:", idx_path)
        try:
            faiss.write_index(index, idx_path)
        except Exception as e:
            print("Error al guardar FAISS:", e)
    else:
        print("--no-faiss activado, se omitió guardado de índice FAISS.")
    print("Guardando metadatos en:", meta_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print("Guardando embeddings en:", npy_path)
    np.save(npy_path, embeddings)
    print("Indexación completada.")


if __name__ == "__main__":
    main()
