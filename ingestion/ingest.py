"""CLI: python -m ingestion.ingest --path ./docs --collection my_docs"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG corpus")
    parser.add_argument("--path", required=True, help="File or directory to ingest")
    parser.add_argument("--collection", default="default", help="Collection name")
    parser.add_argument("--glob", default="**/*.txt", help="Glob pattern for directories")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    # Lazy import so we can show an error before heavy imports
    try:
        import sys, os
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from config import Config
        from ingestion.pipeline import ingest
    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        sys.exit(1)

    cfg = Config()

    console.print(f"Ingesting [bold]{args.path}[/bold] → collection [cyan]{args.collection}[/cyan]")

    with console.status("Embedding and indexing…"):
        n = ingest(
            path=args.path,
            collection=args.collection,
            lance_db_path=cfg.lance_db_path,
            bm25_base_path=cfg.bm25_index_path,
            embed_model=cfg.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            glob=args.glob,
        )

    console.print(f"[green]Done.[/green] Added {n} chunks to '{args.collection}'.")


if __name__ == "__main__":
    main()
