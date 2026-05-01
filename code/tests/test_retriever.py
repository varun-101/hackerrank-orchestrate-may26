import os
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import _clean, _chunk_text, retrieve

def test_clean():
    raw = "---\ntitle: test\n---\n\nActual content\n\n\n\nMore content"
    cleaned = _clean(raw)
    assert "title: test" not in cleaned
    assert "Actual content" in cleaned
    assert "\n\n\n" not in cleaned
    print("[PASS] test_clean")

def test_chunk_text():
    # Create text that will definitely be split
    # CHUNK_CHARS is 2048
    text = "Line 1\n" + "x" * 2000 + "\nLine 2\n" + "y" * 2000
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    assert "Line 1" in chunks[0]
    print(f"[PASS] test_chunk_text ({len(chunks)} chunks)")

def test_retrieve_behavior():
    # Should return empty list safely if index not built
    res = retrieve("how to regenrate my credentials", "HackerRank")
    assert isinstance(res, list)
    print(res)
    print(f"[PASS] test_retrieve_behavior (returned list of size {len(res)})")

if __name__ == "__main__":
    print("=== Retriever Tests ===\n")
    try:
        test_clean()
        test_chunk_text()
        test_retrieve_behavior()
        print("\nAll retriever tests completed.")
    except Exception as e:
        print(f"\n[FAIL] Tests failed with error: {e}")
        sys.exit(1)
