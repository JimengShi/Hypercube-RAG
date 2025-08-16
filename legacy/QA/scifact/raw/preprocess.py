import json
import os
from collections import defaultdict


def build_doc_id_mapping(corpus_path):
    """Return mapping from old doc_id to new sequential id (starting at 1)."""
    mapping = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                old_id = record.get("doc_id")
                if old_id is not None:
                    mapping[int(old_id)] = idx
            except json.JSONDecodeError:
                continue
    return mapping


def process_claims(claims_dev_path, mapping):
    """Process claims_dev.jsonl and return list of formatted dict entries."""
    processed = []
    with open(claims_dev_path, "r", encoding="utf-8") as fin:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue

            evidence = item.get("evidence", {})
            if not evidence:
                # Skip entries without evidence
                continue

            # Collect labels from evidence structure
            labels_set = set()
            for doc_dict in evidence.values():
                for ev in doc_dict:
                    label = ev.get("label")
                    if label:
                        labels_set.add(label)

            if not labels_set:
                # No labels found; skip this claim
                continue
            # Per specification, take first label; warn if multiple
            answers = [next(iter(labels_set))]
            if len(labels_set) > 1:
                print(
                    f"Warning: multiple labels {labels_set} found for claim id {item.get('id')}; using {answers[0]}")

            # Map cited_doc_ids -> new ids
            cited_old = item.get("cited_doc_ids", [])
            relevant_doc = []
            for old_id in cited_old:
                new_id = mapping.get(int(old_id))
                if new_id is not None:
                    relevant_doc.append(new_id)
                else:
                    print(f"Warning: old doc_id {old_id} not found in corpus mapping; skipping.")

            # Skip claims with no relevant documents after mapping
            if not relevant_doc:
                print(f"Warning: no relevant documents found for claim id {item.get('id')}; skipping.")
                continue
                

            processed.append({
                "index": None,
                "question": item.get("claim", ""),
                "answers": answers,
                "relevant_doc": relevant_doc
            })

    # Reindex starting from 1
    for new_idx, entry in enumerate(processed, start=1):
        entry["index"] = new_idx
    return processed


def main():
    corpus_path = os.path.join("corpus", "scifact", "raw", "corpus.jsonl")
    claims_dev_path = os.path.join("QA", "scifact", "raw", "claims_dev.jsonl")
    output_path = os.path.join("QA", "scifact", "claims.json")

    mapping = build_doc_id_mapping(corpus_path)
    processed_claims = process_claims(claims_dev_path, mapping)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write JSON list with indentation for readability
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(processed_claims, fout, ensure_ascii=False, indent=2)

    print(f"Converted {len(processed_claims)} claims saved to {output_path}")


if __name__ == "__main__":
    main() 