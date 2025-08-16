import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import argparse
import os
import json
import time
import re
from together import Together
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ================ read corpus ================
print('Reading the corpus file ...')
with open('corpus/geography/SciDCC-Geography.txt') as f:  # 'corpus/hurricane/SciDCC-Hurricane.txt', 'corpus/geography/SciDCC-Geography.txt', 'corpus/aging_dam/corpus.txt'
    readin = f.readlines()
    corpus = [line.strip() for line in tqdm(readin)]

tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)


# ================ retrive_bm25 ================
def retrive_bm25(query, k):
    start_time = time.time()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_doc_ids = np.argsort(scores)[::-1][:k]
    retrieved_docs = '\n\n'.join([f"Document {idx + 1}: {corpus[doc_id]}" for idx, doc_id in enumerate(top_doc_ids)])
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time: {total_time}")

    return retrieved_docs



# ================ LLMs ================
def llm_answer(model_type, query, k=3):
    """
    @step 1: retrieve
    @step 2: input retrieved docs + instructions to llm
    """
    # retrieved documents input to llm
    docs = retrive_bm25(query, k)

    # set up instruction
    instruction = 'Answer the query based on the given retrieved documents. ' \
    'If the query asks the quantitative analysis, such as starting with "How many", "How much", "How much greater", "How wide", "What percentage", "How far", "How long", "How old", "What portion", "What depth", you must directly output the quantitative answer as short as possible.' \
    'If the query starting with "what percentage" or the query includes the word "likelihood", please directly output the number with %. ' \
    'If the query starting with "what specific", "which years", "who", please directly output the answer without explanation or other information.' \
    'Otherwise, please provide as precise information as possible based the retrieved documents. ' \
    'Documents:\n'

    # set up the LLM
    if model_type == 'gpt-4o':
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": instruction + docs},
                {
                    "role": "user",
                    "content": 'Query: ' + query + '\nAnswer:'
                }
            ]
        )
        res = completion.choices[0].message.content
        return res
    
    elif model_type == 'gpt-4':
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instruction + docs},
                {
                    "role": "user",
                    "content": 'Query: ' + query + '\nAnswer:'
                }
            ]
        )
        res = completion.choices[0].message.content
        return res
    
    elif model_type == 'deepseek':
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEYY" 
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",  
            messages=[
                {"role": "system", "content": instruction + docs},
                {"role": "user", "content": 'Query: ' + query + 'you must show the answer starting with' + '\nAnswer:'}
            ],
            temperature=0.6,
            extra_body={
                "include_reasoning": True
            },
            max_tokens=1024
        )
        res = completion.choices[0].message.content  
        # Remove content between <think> tags (including the tags themselves)
        cleaned_res = re.sub(r'<think>.*?</think>', '', res, flags=re.DOTALL).strip()
        cleaned_res = cleaned_res.split("Answer:")[-1].strip()
        return cleaned_res
    
    elif model_type == 'qwen':
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEYY"  
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo", 
            messages=[
                {"role": "system", "content": instruction + docs},
                {"role": "user", "content": 'Query: ' + query + '\nAnswer:'}
            ]
        )
        res = completion.choices[0].message.content  
        return res    
    
    elif model_type == 'llama3':
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEYY" 
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
            messages=[
                {"role": "system", "content": instruction + docs},
                {"role": "user", "content": 'Query: ' + query + '\nAnswer:'}
            ]
        )
        res = completion.choices[0].message.content  
        return res



# ================ Load QA Dataset ================
def load_dataset(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")




# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--data", type=str, required=True, help="Path to the QA dataset (JSON or CSV).")
    parser.add_argument("--model", type=str, required=True, choices=["gpt-4", "gpt-4o", "deepseek", 'llama3', "qwen"], help="Select llm to get answer.")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved documents.")
    parser.add_argument("--save", type=str, default="false", choices=["true", "false"], help="Evaluation metric: one score or multiple scores.")
    return parser.parse_args()



# ================ Main Function ================
def main():
    args = parse_args()
    
    data_path = f"QA/{args.data}/synthetic_qa.json"

    # Load dataset
    qa_samples = load_dataset(data_path)

    # Initialize evaluation metrics
    llm_output = []

    for i, sample in enumerate(qa_samples):
        question = sample["question"]
        gold_answer = sample["answer"]

        print(f"Q{i+1}: {question}")
        
        predicted_answer = llm_answer(args.model, question, args.k)
        print(f">>> Predicted: {predicted_answer}")
        print(f">>> Ground Truth: {gold_answer}")
        print(f"\n")


        # Save output for each sample
        llm_output.append({
            "index": i+1,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer
        })


    output_dir = f"output/{args.data}/{args.model}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.save == "true":
        with open(f"{output_dir}/llm_output_bm25.json", "w", encoding="utf-8") as f:
            json.dump(llm_output, f, indent=2, ensure_ascii=False)

        print("Saved llm output!")


if __name__ == "__main__":
    main()


