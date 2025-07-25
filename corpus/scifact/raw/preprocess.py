import json
import os


def main():
    """Read corpus.jsonl, concatenate title and abstract sentences, and write one document per line."""
    input_path = os.path.join("corpus", "scifact", "raw", "corpus.jsonl")
    output_path = os.path.join("corpus", "scifact", "pubmed_abstract.txt")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as infile, \
            open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            # Parse JSON line
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines gracefully
                continue

            title = record.get("title", "").strip()
            abstract_sentences = record.get("abstract", [])

            # Some abstracts might be a single string instead of list
            if isinstance(abstract_sentences, list):
                abstract_text = " ".join(sentence.strip() for sentence in abstract_sentences)
            else:
                abstract_text = str(abstract_sentences).strip()

            # Combine title and abstract text
            document = f"{title} {abstract_text}".strip()
            outfile.write(document + "\n")


if __name__ == "__main__":
    main()
