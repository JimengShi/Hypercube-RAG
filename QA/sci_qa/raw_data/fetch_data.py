from datasets import load_dataset

dataset = load_dataset("orkg/SciQA")

for split in ["train", "validation", "test"]:
    data = dataset[split]
    data.to_json(f"QA/sci_qa/raw_data/sciqa_{split}.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"Saved {split} set to sciqa_{split}.jsonl")