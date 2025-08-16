import os
import json
from typing import Dict, List

"""
This script converts the original LegalBench dataset into the desired format.

Steps:
1.  Build a corpus mapping: iterate over `corpus/legalbench/raw/<split>/*.txt`, assign a
global unique ID (starting at 1) to every document, flatten each document so it
   occupies exactly one line, and write the collection for each split to
   `corpus/legalbench/<split>.txt`.

2.  Convert every QA json under `QA/legalbench/raw/*.json` (except this script)
    into the new format and store it at `QA/legalbench/<same-name>.json`.

Output QA format
----------------
[
  {
    "index": 1,
    "question": "...",
    "answers": ["...", "..."],
    "relevant_doc": [3, 42]
  },
  ...
]
"""

BASE_CORPUS_RAW = os.path.join("corpus", "legalbench", "raw")
BASE_CORPUS_OUT = os.path.join("corpus", "legalbench")
BASE_QA_RAW = os.path.join("QA", "legalbench", "raw")
BASE_QA_OUT = os.path.join("QA", "legalbench")

# ---------------------------------------------------------------------------
# Corpus processing
# ---------------------------------------------------------------------------

def _flatten(text: str) -> str:
    """Replace internal newlines with spaces so the whole document is one line."""
    return " ".join(text.split())


def build_corpus_mapping() -> Dict[str, int]:
    """Walk through all raw corpus splits and build a mapping `rel_path -> id`.

    Also writes the flattened corpus for every split.
    """
    mapping: Dict[str, int] = {}

    # Ensure output directory exists
    os.makedirs(BASE_CORPUS_OUT, exist_ok=True)

    # Iterate alphabetically for reproducibility
    for split in sorted(os.listdir(BASE_CORPUS_RAW)):
        split_dir = os.path.join(BASE_CORPUS_RAW, split)
        if not os.path.isdir(split_dir):
            continue

        out_lines: List[str] = []
        print(f"Processing split '{split}' …")

        # Per-split counter starts at 1
        local_id = 1

        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".txt"):
                continue

            rel_path = f"{split}/{fname}"
            full_path = os.path.join(split_dir, fname)

            # Read & flatten document
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = _flatten(f.read())
            except Exception as exc:
                raise RuntimeError(f"Failed reading {full_path}: {exc}") from exc

            # Store mapping & content (per-split IDs starting at 1)
            mapping[rel_path] = local_id
            out_lines.append(content)
            local_id += 1

        # Write one-line-per-doc file for the split
        out_path = os.path.join(BASE_CORPUS_OUT, f"{split}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
        print(f"  -> wrote {len(out_lines)} docs to {out_path}")

    # No single total count printed now because IDs reset per split
    return mapping

# ---------------------------------------------------------------------------
# QA JSON processing
# ---------------------------------------------------------------------------

def convert_qa_files(mapping: Dict[str, int]):
    """Convert every QA json in BASE_QA_RAW and write to BASE_QA_OUT."""

    os.makedirs(BASE_QA_OUT, exist_ok=True)

    for fname in os.listdir(BASE_QA_RAW):
        if not fname.endswith(".json"):
            continue  # skip script or other files

        in_path = os.path.join(BASE_QA_RAW, fname)
        out_path = os.path.join(BASE_QA_OUT, fname)
        print(f"Converting {in_path} …")

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tests = data.get("tests", [])
        converted: List[dict] = []

        for idx, item in enumerate(tests, start=1):
            question = item.get("query", "").strip()
            snippets = item.get("snippets", [])

            answers: List[str] = []
            doc_ids: List[int] = []

            for snip in snippets:
                # Collect answer text
                answer_text = snip.get("answer", "").strip()
                if answer_text:
                    answers.append(answer_text)

                # Map file_path to doc id
                fp = snip.get("file_path")
                if fp is None:
                    continue
                doc_id = mapping.get(fp)
                if doc_id is None:
                    # File might reside under a different rel path; warn once.
                    print(f"WARNING: No mapping found for '{fp}'.")
                    continue
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)

            converted.append({
                "index": idx,
                "question": question,
                "answers": answers,
                "relevant_doc": doc_ids,
            })

        # Write converted file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"  -> wrote {len(converted)} items to {out_path}")

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    mapping = build_corpus_mapping()
    convert_qa_files(mapping)


if __name__ == "__main__":
    main()
