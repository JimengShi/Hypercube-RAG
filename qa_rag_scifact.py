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


def embed_string(target_str, emb_model, dict_path='ent2emb/ent2emb_scifact_latest.pkl', ):
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
with open('corpus/scifact/pubmed_abstract.txt') as f:
    readin = f.readlines()
    corpus = [line.strip() for line in tqdm(readin)]


# ================ build embedding index ================
model = SentenceTransformer('intfloat/e5-base-v2')


# ================ construct hypercube ================
print('Reading the hypercube files ...')


dimensions = ['date', 'location', 'person', 'quantity', 'organizations_research_initiatives', 'medicine_health', 'genetics_biology', 'immunology_neuroscience', 'pharmacology']#, 'chemistry']
hypercube = {'date': defaultdict(list),
             'location': defaultdict(list),
             'person': defaultdict(list),
             'quantity': defaultdict(list),
             'organizations_research_initiatives': defaultdict(list),
             'medicine_health': defaultdict(list),
             'genetics_biology': defaultdict(list),
             'immunology_neuroscience': defaultdict(list),
             'pharmacology': defaultdict(list),}


for dimension in dimensions:
    with open(f'hypercube_new/scifact/{dimension}.txt') as f:
        readin = f.readlines()
        for i, line in tqdm(enumerate(readin), total=len(readin), desc=f"{dimension}"):
            tmp = json.loads(line)
            for k in tmp:
                hypercube[dimension][k].append(i)
                
                embed_string(target_str=k, emb_model=model, dict_path='ent2emb/ent2emb_scifact_latest.pkl', )



def get_docs_from_cells(cells, top_k):
    """
    cells: query extraction
    """
    if cells is None: return []
    tmp_ids = []
    
    for k, v in cells.items():
        assert k in hypercube
        
        for vv in v:
            vv = vv.lower()
            if vv in hypercube[k]:
                tmp_ids.extend(hypercube[k][vv])
            else:
                vv_emb = model.encode(vv, normalize_embeddings=True)
                
                for cand in hypercube[k]:
                    if ent2emb[cand] @ vv_emb > 0.9: 
                        tmp_ids.extend(hypercube[k][cand])

    doc_ids = topk_most_freq_id(tmp_ids, top_k)

    return list(doc_ids)


# ================ LLMs ================
def llm_answer(model_type, query, cells, k, retrieval_method='hypercube'):
    assert model_type in ['gpt-4o-mini', 'gpt-4o', 'llama3']
    assert retrieval_method in ['hypercube', 'semantic', 'union']


    ### set up hypercube retriever 
    structure_docs = get_docs_from_cells(cells, k)

    
    if retrieval_method == 'hypercube':
        doc_ids = structure_docs
        print(f'>>> Doc ids by hypercube retrieval: {[id + 1 for id in doc_ids]} \n')

    # limit retrieved documents of hypercube as input to llm
    docs = '\n\n'.join([f"Document {idx + 1}: {corpus[doc_id]}" for idx, doc_id in enumerate(doc_ids)])


    # set up instruction
    instruction = 'Determine if the claim in the query is SUPPORT or CONTRADICT, based on the given retrieved documents. ' \
    'Please keep the answer as precise as possible. You must output either SUPPORT or CONTRADICT' \
    'Documents:\n'

    # set up the LLM
    if model_type == 'llama3':
        os.environ["TOGETHER_API_KEY"] = "your TOGETHER_API_KEY"  # or set it externally
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
            messages=[
                {"role": "system", "content": instruction + docs},
                {"role": "user", "content": 'Query: ' + query + '\nAnswer:'}
            ]
        )
        res = completion.choices[0].message.content  
        return res, [id + 1 for id in doc_ids]
    
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
        return res, [id + 1 for id in doc_ids]


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
        f"date dimension can be Time, Period, Duration, Period, Hour, Minutes, Day, Week, Month, Year entities/phrases.\n"
        f"location dimension can be geographic locations-related, nationality-related, city-related, state-related, prinvince-related entities/phrases/numbers/zipcodes.\n"
        f"person dimension can be related to names of people, males, females, children, adults and the descriptive phrases/terms such as person, women, adult, and the role name such as patient, doctor, lawer.\n"
        f"quantity dimension can be quantity-related, such as Numbers, Percentage, Ratio, Rates, Proportions, Fractions, Ages, Weights, Measurements, Finance amounts, Money amounts entities/phrases/numbers.\n"        
        f"organizations_research_initiatives dimension can be organization-related, medical research, cancer research, health initiatives, health surveys, disease studies-related entities/phrases.\n"
        f"medicine_health dimension can be medical, cancer, cardiology, diseases terms healty-related entities/phrases.\n"
        f"genetics_biology dimension can be genetics-related, biology-related, biochemistry-related, molecular biology-related, cell biology-related entities/phrases/terms.\n"
        f"immunology_neuroscience dimension can be immunology-related and neuroscience-related entities/phrases/terms.\n"
        f"pharmacology dimension can be pharmacology-related, pharmaceutical sciences-related entities/phrases/terms.\n"
        f"chemistry dimension can be chemistry-related, biochemistry-related entities/phrases/terms.\n"        
        f"For each of the above dimension, synthesize queries that are informative, self-complete, and mostly likely to retrieve target documents for answering the question.\n"
        f"Note that each of your query should be an entity or a short phrase and its associated dimension.\n\n"

        f"Example Input:\n"
        f"Question: 1/2000 in UK have abnormal PrP positivity\n"
        f"Example Output:\n"
        f"Query 1:\n"
        f"query_dimension: 'medicine_health'; query_content: 'abnormal PrP positivity';\n"
        f"Query 2:\n"
        f"query_dimension: 'quantity'; query_content: '1/2000';\n"
        f"Query 3:\n"
        f"query_dimension: 'location'; query_content: 'UK';\n"
    )

    input_prompt = (
        f"Question: {query}"
    )
    
    class Query(BaseModel):
        query_content: str = Field(
            ...,
            title='Entity or phrase to query the documents'
        )
        query_dimension: Literal['date', 'location', 'person', 'quantity', 'organizations_research_initiatives', 'medicine_health', 'genetics_biology', 'immunology_neuroscience', 'pharmacology'] = Field(
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
            embed_string(target_str=ent.query_content, emb_model=model, dict_path='ent2emb/ent2emb_scifact_latest.pkl', )
        return cells
    
    except Exception as e:
        print(str(e))
        return None


# ================ Evaluation Metrics ================
from utils.metric import bleu_score, semantic_score, f1_score
from utils.metric import precision_at_k, recall_at_k


# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--data", type=str, required=True, help="Path to the QA dataset (JSON or CSV).")
    parser.add_argument("--model", type=str, required=True, choices=["gpt-4o-mini", "gpt-4o", 'llama3'], help="Select llm to get answer.")
    parser.add_argument("--retrieval_method", type=str, required=True, default="hypercube", choices=['hypercube', 'semantic', 'union'], help="Retrieval methods.")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved documents.")
    parser.add_argument("--save", type=str, default="false", choices=["true", "false"], help="Evaluation metric: one score or multiple scores.")
    return parser.parse_args()



# ================ Main Function ================
def main():
    args = parse_args()

    data_path = f"QA/{args.data}/claims.json"

    # Load dataset
    qa_samples = load_dataset(data_path)

    # Initialize evaluation metrics
    llm_output = []

    # Initialize evaluation metrics
    total_bleu, total_semantic, total_f1 = 0, 0, 0
    total_precison, total_recall = 0.0, 0.0

    for i, sample in enumerate(qa_samples):
        question = sample["question"]
        gold_answer = " ".join(sample["answers"])
        relevant_docs = sample["relevant_doc"]

        print(f"Q{i+1}: {question}")

        cells = decompose_query(query=question)
        print(f">>> Identified cells from the query: {cells} \n")
        
        predicted_answer, return_doc_ids = llm_answer(args.model, question, cells, args.k, args.retrieval_method)
        print(f">>> Predicted: {predicted_answer}")
        print(f">>> Ground Truth: {gold_answer}")
        print(f">>> return_doc_ids: {return_doc_ids}")
        print(f">>> relevant_docs: {relevant_docs}")
        print(f'Semantic_score: {semantic_score(predicted_answer, gold_answer):.4f}')
        print(f"\n")

        total_bleu += bleu_score(predicted_answer, gold_answer)
        total_semantic += semantic_score(predicted_answer, gold_answer)
        total_f1 += f1_score(predicted_answer, gold_answer)

        if len(return_doc_ids) == 0:
            return_doc_ids = [0]
        total_precison += precision_at_k(return_doc_ids, relevant_docs)
        total_recall += recall_at_k(return_doc_ids, relevant_docs)

        # Save output for each sample
        llm_output.append({
            "index": i+1,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "f1_score": f1_score(predicted_answer, gold_answer),
            "semantic_score": semantic_score(predicted_answer, gold_answer),
            "return_doc_ids": return_doc_ids,
            "relevant_docs": relevant_docs,
            "precision": precision_at_k(return_doc_ids, relevant_docs),
            "recall": recall_at_k(return_doc_ids, relevant_docs)
        })


    # Averaging Summary Information
    num_samples = len(qa_samples)
    print(f"Total samples: {num_samples}, average F1: {total_f1 / num_samples:.4f}, average Semantic: {total_semantic / num_samples:.4f}, average precision: {total_precison / num_samples:.4f}, average reall: {total_recall / num_samples:.4f}")

    

    # Assume you have these variables already defined
    results = {
        "total_samples": num_samples,
        "average_F1": total_f1 / num_samples,
        "average_Semantic": total_semantic / num_samples, 
        "average_Precision": total_precison / num_samples,
        "average_Recall": total_recall / num_samples
    }


    output_dir = f"output/{args.data}/{args.model}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.save == "true":
        with open(f"{output_dir}/llm_output_{args.retrieval_method}_k{args.k}_responses.json", "w", encoding="utf-8") as f:
            json.dump(llm_output, f, indent=4, ensure_ascii=False)

        with open(f"{output_dir}/llm_output_{args.retrieval_method}_k{args.k}_scores.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print("Saved llm output!")


if __name__ == "__main__":
    main()

