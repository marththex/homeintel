"""
scripts/test_run_stream.py — Smoke test for RAGChain.run_stream (needs Qdrant + Ollama).

Run from the backend directory:
    cd backend
    python ../scripts/test_run_stream.py
Exit 0 = pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from rag.chain import get_chain


def main():
    chain = get_chain()

    events = list(chain.run_stream("What companies has Jane Doe worked for?"))

    tags = [t for t, _ in events]
    assert tags[0] == "sources", f"first event must be 'sources', got {tags[:3]}"
    assert tags[-1] == "done", f"last event must be 'done', got {tags[-3:]}"
    assert "token" in tags, "expected at least one token event"

    answer = "".join(p for t, p in events if t == "token")
    assert answer.strip(), "streamed answer is empty"

    # run() still returns the same shape
    result = chain.run("What companies has Jane Doe worked for?")
    assert set(result.keys()) == {"answer", "docs"}, result.keys()
    assert isinstance(result["answer"], str) and result["answer"].strip()

    print("PASS — sources→token→done, run() intact")


if __name__ == "__main__":
    main()
