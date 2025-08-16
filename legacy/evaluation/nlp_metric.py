import json
import argparse
import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ================ Load Dataset ================
def load_dataset(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            return json.load(f)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        return df.to_dict(orient="records")  # Converts to list of dicts
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")
    

# ================ Evaluation Metrics ================
from utils.metric import exact_match, f1_score, bleu_score, rouge_score, semantic_score


# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--data", type=str, required=True, default="hurricane", choices=['hurricane', 'geography', 'aging_dam'], help="data set.")
    parser.add_argument("--model", type=str, required=True, choices=["gpt-4", "gpt-4o", "gpt-3.5-turbo", "deepseek", 'llama3', 'llama4', 'gemma', "qwen"], help="Select llm to get answer.")
    parser.add_argument("--retrieval_method", type=str, default="hypercube", choices=['hypercube', 'semantic', 'union', 'hipporag', 'contriever', 'graphrag', 'bm25', 'none'], help="Retrieval methods.")
    parser.add_argument("--metric", type=str, default="all", choices=["em", "f1", "bleu", "rouge", "semantic", "all"], help="Evaluation metric: one score or multiple scores.")
    return parser.parse_args()


# ================ Main Function ================
def main():
    args = parse_args()

    if args.retrieval_method == 'none':
        llm_output_dir = f"output/{args.data}/{args.model}/llm_output_no_rag.json"
    elif args.retrieval_method == 'bm25':
        llm_output_dir = f"output/{args.data}/{args.model}/llm_output_bm25.json"
    elif args.retrieval_method == 'contriever':
        llm_output_dir = f"output/{args.data}/{args.model}/llm_output_semantic_contriever.json"
    elif args.retrieval_method == 'graphrag':
        llm_output_dir = f"output/{args.data}/{args.model}/llm_output_graphrag.json"
    elif args.retrieval_method == 'hipporag':
        llm_output_dir = f"output/{args.data}/{args.model}/llm_output_hipporag.json"
    else:
        llm_output_dir = f"output/{args.data}/{args.model}/llm_output_{args.retrieval_method}.json"


    # Load dataset
    qa_samples = load_dataset(llm_output_dir)

    # Evaluation
    print("\n--- QA Evaluation ---\n")

    # Initialize evaluation metrics
    total_em, total_f1, total_bleu, total_rouge, total_semantic = 0, 0, 0, 0, 0

    for i, sample in enumerate(qa_samples):
        print(f"Sample: {i+1}")
        gold_answer = sample["gold_answer"]  
        predicted_answer = sample["predicted_answer"]

        # print(f">>> Predicted: {predicted_answer}")
        # print(f">>> Ground Truth: {gold_answer}")
        
        # Calculate metrics of each sample based on the selected option
        if args.metric in ["em", "all"]:
            em = exact_match(predicted_answer, gold_answer)
            total_em += em
            # print(f"Exact Match: {em:.2f}")
        if args.metric in ["f1", "all"]:
            f1 = f1_score(predicted_answer, gold_answer)
            total_f1 += f1
            # print(f"F1 Score: {f1:.2f}")
        if args.metric in ["bleu", "all"]:
            bleu = bleu_score(predicted_answer, gold_answer)
            total_bleu += bleu
            # print(f"Bleu Score: {bleu:.2f}")
        if args.metric in ["rouge", "all"]:
            rouge = rouge_score(predicted_answer, gold_answer)
            total_rouge += rouge
            # print(f"ROUGE Score: {rouge:.2f}")
        if args.metric in ["semantic", "all"]:
            semantic = semantic_score(predicted_answer, gold_answer)
            total_semantic += semantic
            # print(f"Semantic Score: {semantic:.2f}")


    # Averaging Summary Information
    num_samples = len(qa_samples)
    print(f"Total samples: {num_samples}, average EM: {total_em / num_samples:.4f}, average F1: {total_f1 / num_samples:.4f}, average BLEU: {total_bleu / num_samples:.4f}, average ROUGE: {total_rouge / num_samples:.4f}, average Semantic: {total_semantic / num_samples:.4f}")


    # if args.metric in ["em", "all"]:
    #     print(f"Average EM: {total_em / num_samples:.4f}")
    # if args.metric in ["f1", "all"]:
    #     print(f"Average F1: {total_f1 / num_samples:.4f}")
    # if args.metric in ["bleu", "all"]:
    #     print(f"Average BLEU: {total_bleu / num_samples:.4f}")
    # if args.metric in ["rouge", "all"]:
    #     print(f"Average ROUGE: {total_rouge / num_samples:.4f}")
    # if args.metric in ["semantic", "all"]:
    #     print(f"Average Semantic: {total_semantic / num_samples:.4f}")


if __name__ == "__main__":
    main()
