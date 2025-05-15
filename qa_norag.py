from openai import OpenAI
import json
import argparse
import pandas as pd
from together import Together
import os
import re

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


# ================ LLMs ================
def llm_answer(question, instruction, model_type):
    if model_type == 'llama3':
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEY"  
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ]
        )
        res = completion.choices[0].message.content  
        return res
    
    elif model_type == 'gemma':
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEY"  
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="meta-llama/gemma-2b-it", 
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ]
        )
        res = completion.choices[0].message.content  
        return res
    
    elif model_type == 'deepseek':
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEY"  
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",  
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            extra_body={
                "include_reasoning": True
            },
            max_tokens=4096
        )
        res = completion.choices[0].message.content  
        # Remove content between <think> tags (including the tags themselves)
        cleaned_res = re.sub(r'<think>.*?</think>', '', res, flags=re.DOTALL).strip()
        cleaned_res = cleaned_res.split("Answer:")[-1].strip()
        return cleaned_res
    
    elif model_type == 'qwen':
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEY"  # or set it externally
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo", 
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ]
        )
        res = completion.choices[0].message.content  
        return res
    
    elif model_type == "gpt-4":
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_type,  
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    
    elif model_type == "gpt-4o":
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_type,  
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    
    elif model_type == "gpt-3.5-turbo":
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_type, 
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        return response.choices[0].message.content



# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--model", type=str, required=True, choices=["gpt-4", "gpt-4o", "gpt-3.5-turbo", "llama3", "llama4", "deepseek", "qwen"], help="Select llm to get answer.")
    parser.add_argument("--data", type=str, required=True, help="Path to the QA dataset (JSON or CSV).")
    parser.add_argument("--save", type=str, default="true", choices=["true", "false"], help="Evaluation metric: one score or multiple scores.")
    return parser.parse_args()


# ================ Main Function ================
def main():
    args = parse_args()
    
    data_path = f"QA/{args.data}/synthetic_qa.json"

    # Load dataset
    qa_samples = load_dataset(data_path)
    
    # Set up instruction for the model
    instruction = "You are a helpful question answering assistant. Please output the answers as short as possible without punctuation."
    
    # Evaluation
    print("\n--- QA Evaluation ---\n")
    
    llm_output = []
    for i, sample in enumerate(qa_samples):
        question = sample["question"]
        gold_answer = sample["answer"]
        
        print(f"Q{i+1}: {question}")
        predicted_answer = llm_answer(question, instruction, args.model)   #["gpt-4", "gpt-3.5-turbo"]
        print(f"Predicted: {predicted_answer}")
        print(f"Ground Truth: {gold_answer}")
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
        with open(f"{output_dir}/llm_output_no_rag.json", "w", encoding="utf-8") as f:
            json.dump(llm_output, f, indent=2, ensure_ascii=False)

        print("Saved llm output!")


if __name__ == "__main__":
    main()