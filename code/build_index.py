"""
Offline indexing script — run this once before executing the pipeline.

Usage:
    python build_index.py                     # index all three domains
    python build_index.py --domain HackerRank # index one domain
    python build_index.py --force             # rebuild even if index exists
"""

import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from retriever import build_index, DOMAIN_DIRS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build domain-sharded retrieval indices")
    parser.add_argument(
        "--domain",
        choices=list(DOMAIN_DIRS.keys()),
        help="Index a single domain (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild index even if it already exists",
    )
    args = parser.parse_args()

    domains = [args.domain] if args.domain else None
    build_index(domains=domains, force=args.force)
    print("\nIndexing complete.")
