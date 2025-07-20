import os
import json
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
import heapq
from openai import OpenAI
import pickle
import argparse
from utils.topk_most_freq import topk_most_freq_id
from together import Together
import os
import re
import time

# from IPython import embed
ent2emb = None


def embed_string(target_str, emb_model, dict_path='./ent2emb.pkl', ):
    global ent2emb
    if ent2emb is None:
        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                ent2emb = pickle.load(f)      
        else:
            ent2emb = {}
            print(f"No saved dictionary found, initialized a new dict from string to corresponding embedding.")

    if target_str in ent2emb:
        return 
    else:
        ent2emb[target_str] = emb_model.encode('query: ' + target_str, normalize_embeddings=True)
        with open(dict_path, 'wb') as f:
            pickle.dump(ent2emb, f)


# ================ Load Dataset ================
def load_dataset(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")
    

# ================ read corpus ================
print('Reading the corpus file ...')
with open('corpus/hurricane/SciDCC-Hurricane.txt') as f:
    readin = f.readlines()
    corpus = [line.strip() for line in tqdm(readin)]


# ================ build embedding index ================
model = SentenceTransformer('intfloat/e5-base-v2')
# model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
doc_embeddings = model.encode(['passage: ' + text for text in corpus], normalize_embeddings=True)


# ================ construct hypercube ================
print('Reading the hypercube files ...')

dimensions = ['location', 'person', 'event', 'organization', 'theme', 'date']
hypercube = {'location': defaultdict(list),
             'event': defaultdict(list),
             'date': defaultdict(list),
             'organization': defaultdict(list),
             'person': defaultdict(list),
             'theme': defaultdict(list),}


for dimension in dimensions:
    with open(f'hypercube/hurricane/{dimension}.txt') as f:
        readin = f.readlines()
        for i, line in tqdm(enumerate(readin), total=len(readin), desc=f"{dimension}"):
            tmp = json.loads(line)
            for k in tmp:
                hypercube[dimension][k].append(i)
                
                embed_string(target_str=k, emb_model=model, dict_path='./ent2emb.pkl', )


def get_docs_from_cells(cells):
    if cells is None: return []
    tmp_ids = []
    # doc_ids = set(list(range(len(corpus))))
    for k, v in cells.items():
        assert k in hypercube
        
        for vv in v:
            # print("vv:", vv)
            if vv in hypercube[k]:
                tmp_ids.extend(hypercube[k][vv])
            else:
                vv_emb = ent2emb[vv]
                for cand in hypercube[k]:
                    if ent2emb[cand] @ vv_emb > 0.9: 
                        tmp_ids.extend(hypercube[k][cand])
            # print(">>> tmp_ids 1:", tmp_ids)

    # print(">>> tmp_ids total:", tmp_ids)

    doc_ids = topk_most_freq_id(tmp_ids, 3)

    print(">>> doc_ids:", doc_ids)
                        
    # doc_ids = doc_ids.intersection(set(tmp_ids))
    return list(doc_ids)


# ================ LLMs ================
def llm_answer(model_type, query, cells, k=3, retrieval_method='hypercube'):
    assert model_type in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o', 'deepseek', 'llama3', 'qwen']
    assert retrieval_method in ['hypercube', 'semantic', 'union']


    ### set up retriever searching combination
    # get docs from cells (hypercube return)
    structure_docs = get_docs_from_cells(cells)

    # get query embedding (semantic return)
    query_embedding = model.encode(['query: ' + query])
    scores = np.matmul(doc_embeddings, query_embedding.transpose())[:,0]
    semantic_docs = [index for _, index in heapq.nlargest(k, ((v, i) for i, v in enumerate(scores)))]



    
    if retrieval_method == 'hypercube':
        doc_ids = structure_docs
        print(f'>>> Doc ids by hypercube retrieval: {[id + 1 for id in doc_ids]} \n')
    elif retrieval_method == 'semantic':
        doc_ids = semantic_docs
        print(f'>>> Doc ids by semantic retrieval: {[id + 1 for id in doc_ids]} \n')
    elif retrieval_method == 'union':
        doc_ids = list(set(structure_docs).union(set(semantic_docs)))
        doc_ids = doc_ids[:5]
        print(f'>>> Doc ids by union retrieval: {[id + 1 for id in doc_ids]} \n') 
    
    # limit retrieved documents of hypercube as input to llm
    docs = '\n\n'.join([f"Document {idx + 1}: {corpus[doc_id]}" for idx, doc_id in enumerate(doc_ids)])


    # set up instruction
    instruction = 'Answer the query based on the given retrieved documents. ' \
    'If the query asks the quantitative analysis, such as starting with "How many", "How much", "How much greater", "How wide", "What percentage", "How far", "How long", "How old", "What portion", "What depth", you must directly output the quantitative answer as short as possible.' \
    'If the query starting with "what percentage" or the query includes the word "likelihood", please directly output the number with %. ' \
    'If the query starting with "what specific", "which years", "who", please directly output the answer without explanation or other information.' \
    'Otherwise, please provide as precise information as possible based the retrieved documents. ' \
    'Documents:\n'

    # set up the LLM
    if model_type == 'llama3':
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEY"  
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
    
    elif model_type == 'deepseek':
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEY" 
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
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEY"  # or set it externally
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
    
    elif model_type == 'gpt-4o':
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
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
    
    elif model_type == "gpt-3.5-turbo":
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
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


from pydantic import (
    BaseModel,
    Field
)

from typing import (
    List, 
    Literal
)


def decompose_query(query):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system_prompt = (
        f"You are an expert on question understanding. "
        f"Your task is to:\n"
        f"1. **Comprehend the given question**: understand what the question asks, how to answer it step by step, and all concepts, aspects, or directions that are relevant to each step.\n"
        f"2. **Compose queries to retrieve documents for answering the question**: each document are indexed by the entities or phrases occurred inside and those entities or phrases lie within following dimensions: {dimensions}. "
        f"For each of the above dimension, synthesize queries that are informative, self-complete, and mostly likely to retrieve target documents for answering the question.\n"
        f"Note that each of your query should be an entity or a short phrase and its associated dimension.\n\n"
        f"Example Input:\n"
        f"Question: How Indian Monsoons Influence Atlantic Hurricane Paths?\n"
        f"Example Output:\n"
        f"Query 1:\n"
        f"query_dimension: 'location'; query_content: 'Atlantic';\n"
        f"Query 2:\n"
        f"query_dimension: 'theme'; query_content: 'Indian Monsoons';\n"
        f"Query 3:\n"
        f"query_dimension: 'theme'; query_content: 'Hurricane Paths';\n"
    )
    
    input_prompt = (
        f"Question: {query}"
    )
    
    class Query(BaseModel):
        query_content: str = Field(
            ...,
            title='Entity or phrase to query the documents'
        )
        
        query_dimension: Literal['person', 'theme', 'event', 'location', 'organization', 'date'] = Field(
            ...,
            title='Dimension of the entity or phrase to query documents'
        )
        
    class AllQueries(BaseModel):
        list_of_queries: List[Query] = Field(
            ...,
            title='List of queries following the required format based on the question comprehension'
        )
        
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_prompt}
        ],
        max_tokens=4096,
        temperature=0,
        n=1,
        response_format=AllQueries,
    )

    try:
        detected_ents = response.choices[0].message.parsed
        
        if detected_ents is None or len(detected_ents.list_of_queries) == 0: return None
        cells = defaultdict(list)
        
        for ent in detected_ents.list_of_queries:
            cells[ent.query_dimension].append(ent.query_content)
            embed_string(target_str=ent.query_content, emb_model=model, dict_path='./ent2emb.pkl', )
        return cells
    
    except Exception as e:
        print(str(e))
        return None


# ================ Evaluation Metrics ================
from utils.metric import bleu_score, semantic_score



# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--data", type=str, required=True, help="Path to the QA dataset (JSON or CSV).")
    parser.add_argument("--model", type=str, required=True, choices=["gpt-4", "gpt-4o", "gpt-3.5-turbo", "deepseek", 'llama3', "qwen"], help="Select llm to get answer.")
    parser.add_argument("--retrieval_method", type=str, required=True, default="hypercube", choices=['hypercube', 'semantic', 'union'], help="Retrieval methods.")
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

    # Initialize evaluation metrics
    total_bleu, total_semantic = 0, 0

    for i, sample in enumerate(qa_samples):
        question = sample["question"]
        gold_answer = sample["answer"]

        print(f"Q{i+1}: {question}")

        cells = decompose_query(query=question)
        print(f">>> Identified cells from the query: {cells} \n")
        
        predicted_answer = llm_answer(args.model, question, cells, args.k, args.retrieval_method)
        print(f">>> Predicted: {predicted_answer}")
        print(f">>> Ground Truth: {gold_answer}")
        print(f"\n")

        total_bleu += bleu_score(predicted_answer, gold_answer)
        total_semantic += semantic_score(predicted_answer, gold_answer)

        # Save output for each sample
        llm_output.append({
            "index": i+1,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer
        })


    # Averaging Summary Information
    num_samples = len(qa_samples)
    print(f"Total samples: {num_samples}, average BLEU: {total_bleu / num_samples:.4f}, average Semantic: {total_semantic / num_samples:.4f}")


    output_dir = f"output/hurricane/{args.model}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.save == "true":
        with open(f"{output_dir}/llm_output_{args.retrieval_method}.json", "w", encoding="utf-8") as f:
            json.dump(llm_output, f, indent=2, ensure_ascii=False)

        print("Saved llm output!")


if __name__ == "__main__":
    main()

