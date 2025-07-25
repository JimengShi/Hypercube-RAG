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
import os
from src.api.openai_api import chat
from concurrent.futures import ThreadPoolExecutor

# from IPython import embed
ent2emb = None
# Track whether new embeddings have been added so we can persist once at the end
ent2emb_changed = False

def embed_string(target_str, emb_model, dict_path='ent2emb/ent2emb_hurricane.pkl', ):
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
        # Mark that we need to flush to disk later
        global ent2emb_changed
        ent2emb_changed = True


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


dimensions = ['date', 'event', 'location', 'organization', 'person', 'theme']
hypercube = {'date': defaultdict(list),
             'event': defaultdict(list),
             'location': defaultdict(list),
             'person': defaultdict(list),
             'theme': defaultdict(list),
             'organization': defaultdict(list),}


for dimension in dimensions:
    with open(f'hypercube/hurricane_human/{dimension}.txt') as f:
        readin = f.readlines()
        # readin = readin[:50]
        for i, line in tqdm(enumerate(readin), total=len(readin), desc=f"{dimension}"):
            tmp = json.loads(line)
            for k in tmp:
                hypercube[dimension][k].append(i)
                
                embed_string(target_str=k, emb_model=model, dict_path='ent2emb/ent2emb_hurricane.pkl', )

# ================ Persist ent2emb once after indexing ================

# Flush the ent2emb cache to disk only once to avoid frequent io bottlenecks
if ent2emb is not None and ent2emb_changed:
    ENT2EMB_PATH = 'ent2emb/ent2emb_hurricane.pkl'
    os.makedirs(os.path.dirname(ENT2EMB_PATH), exist_ok=True)
    with open(ENT2EMB_PATH, 'wb') as f:
        pickle.dump(ent2emb, f)
    print(f"Saved ent2emb cache with {len(ent2emb)} entries to {ENT2EMB_PATH}")


def get_docs_from_cells(cells, top_k):
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

    doc_ids = topk_most_freq_id(tmp_ids, top_k)

    # print(">>> doc_ids:", doc_ids)
                        
    # doc_ids = doc_ids.intersection(set(tmp_ids))
    return list(doc_ids)

# ================ Retrieve docs from a plain list of entities (dimension-agnostic) ================

def get_docs_from_entities(entities, top_k):
    """Return document ids (0-indexed) based on entity overlap across *all* dimensions.
    If an entity is not found verbatim, fall back to embedding similarity (>0.9).
    """
    if entities is None:
        return []

    tmp_ids = []
    for ent in entities:
        # Ensure embedding exists for the entity
        embed_string(target_str=ent, emb_model=model, dict_path='ent2emb/ent2emb_hurricane.pkl')

        # Direct match across every dimension
        for dim in dimensions:
            if ent in hypercube[dim]:
                tmp_ids.extend(hypercube[dim][ent])

        # Fallback: embedding similarity search
        ent_emb = ent2emb[ent]
        for dim in dimensions:
            for cand in hypercube[dim]:
                if ent2emb[cand] @ ent_emb > 0.9:
                    tmp_ids.extend(hypercube[dim][cand])

    doc_ids = topk_most_freq_id(tmp_ids, top_k)
    return list(doc_ids)

# ================ Prompt Builder (Batch Friendly) ================

def build_prompt(query, cells, k, retrieval_method='hypercube'):
    """Return a single prompt string for the LLM together with the retrieved doc ids (1-indexed).
    This function contains the same retrieval logic as `llm_answer` but only prepares the
    prompt. Later we can feed a list of such prompts to the high-throughput `chat` API
    in one batch call.
    """
    assert retrieval_method in ['hypercube', 'semantic', 'union', 'none']

    # --- Retrieve docs from hyper-cube structure (only if applicable) ---
    structure_docs = []
    if retrieval_method in ['hypercube', 'union']:
        structure_docs = get_docs_from_cells(cells, k)

    # --- Retrieve docs via semantic similarity ---
    query_embedding = model.encode(['query: ' + query])
    scores = np.matmul(doc_embeddings, query_embedding.transpose())[:, 0]
    semantic_docs = [index for _, index in heapq.nlargest(k, ((v, i) for i, v in enumerate(scores)))]

    if retrieval_method == 'none':
        doc_ids = []
    elif retrieval_method == 'hypercube':
        doc_ids = structure_docs
    elif retrieval_method == 'semantic':
        doc_ids = semantic_docs
    else:  # union
        doc_ids = list(set(structure_docs).union(set(semantic_docs)))
        doc_ids = doc_ids[:k]

    # Build docs context
    docs = '\n\n'.join([f"Document {idx + 1}: {corpus[doc_id]}" for idx, doc_id in enumerate(doc_ids)])

    # Compose final prompt (system prompt will be automatically added by `chat`)
    instruction = (
        'Answer the query based on the given retrieved documents. '
        'Please keep the answer as precise as possible.\nDocuments:\n'
    )

    prompt = f"{instruction}{docs}\n\nQuery: {query}\nAnswer:"

    return prompt, [id + 1 for id in doc_ids]


from pydantic import (
    BaseModel,
    Field
)

from typing import (
    List, 
    Literal
)


def decompose_query(query, seed=42):
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
        seed=seed,
        response_format=AllQueries,
    )

    try:
        detected_ents = response.choices[0].message.parsed
        
        if detected_ents is None or len(detected_ents.list_of_queries) == 0: return None
        cells = defaultdict(list)
        
        for ent in detected_ents.list_of_queries:
            cells[ent.query_dimension].append(ent.query_content)
            embed_string(target_str=ent.query_content, emb_model=model, dict_path='ent2emb/ent2emb_hurricane.pkl', )
        return cells
    
    except Exception as e:
        print(str(e))
        return None

# ================ Evaluation Metrics ================
from utils.metric import bleu_score, semantic_score, f1_score

# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--data", type=str, required=True, help="Path to the QA dataset (JSON or CSV).")
    parser.add_argument("--model", type=str, required=True, choices=["gpt-4o-mini", "gpt-4o", "deepseek", 'llama3', 'llama4', 'gemma', "qwen"], help="Select llm to get answer.")
    parser.add_argument("--retrieval_method", type=str, required=True, default="hypercube", choices=['hypercube', 'semantic', 'union', 'none'], help="Retrieval methods.")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved documents.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for OpenAI chat completions (default 42, always forwarded).")
    parser.add_argument("--save", type=str, default="true", choices=["true", "false"], help="Evaluation metric: one score or multiple scores.")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads for query decomposition (default 8).")
    return parser.parse_args()

# ================ Main Function ================
def main():
    args = parse_args()

    data_path = f"QA/{args.data}/synthetic_qa.json"

    # Load dataset
    qa_samples = load_dataset(data_path)
    # qa_samples = qa_samples[:5]

    # Prepare prompts and supporting evaluation info in batch (query decomposition in parallel)
    questions = [sample["question"] for sample in qa_samples]

    # Decide whether to perform entity extraction based on retrieval method
    if args.retrieval_method in ["hypercube", "union"]:
        print(f"\nDecomposing {len(questions)} queries using {args.num_threads} threads ...\n")

        # Choose the appropriate decomposition function
        decompose_fn = lambda q: decompose_query(q, seed=args.seed)

        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            cells_list = list(
                tqdm(
                    executor.map(decompose_fn, questions),
                    total=len(questions),
                    desc="Decomposing Queries",
                )
            )
    else:
        # For 'semantic' and 'none' retrieval, skip decomposition to save cost
        cells_list = [None] * len(questions)

    # Build prompts sequentially (light-weight) and collect meta information
    prompts: list[str] = []
    doc_ids_all: list[list[int]] = []
    meta_info = []

    for i, sample in enumerate(qa_samples):
        question = sample["question"]
        gold_answer = sample["answer"]

        cells = cells_list[i]
        print(f"Q{i+1}: {question}\n>>> Identified cells: {cells}\n")

        prompt, return_doc_ids = build_prompt(question, cells, args.k, args.retrieval_method)
        print(f">>> Selected doc ids: {return_doc_ids} (retrieval_method={args.retrieval_method})\n")

        prompts.append(prompt)
        doc_ids_all.append(return_doc_ids)
        meta_info.append((question, gold_answer))

    # Call the high-throughput chat API once with all prompts
    # Map CLI model name to the identifier expected by the `chat` helper if necessary
    MODEL_ALIAS = {
        'gpt-4o-mini': 'gpt-4o-mini',
        'gpt-4o': 'gpt-4o-2024-11-20',
        'qwen': 'qwen3-32b',
        # Add more aliases here if needed
    }

    chat_model_name = MODEL_ALIAS.get(args.model)
    if chat_model_name is None:
        raise ValueError(f"Model '{args.model}' is not supported by the parallel chat API.")

    print(f"\nSending {len(prompts)} prompts to chat API '{chat_model_name}' in parallel ...\n")
    chat_kwargs = {"model_name": chat_model_name, "temperature": 1.0, "seed": args.seed}

    responses = chat(prompts, **chat_kwargs)

    # Initialize evaluation metrics
    llm_output = []
    total_bleu = total_semantic = total_f1 = 0.0
    total_precison = total_recall = 0.0

    # Iterate over responses and evaluate
    for i, (response_text, (question, gold_answer), return_doc_ids) in enumerate(zip(responses, meta_info, doc_ids_all)):
        print(f"Q{i+1} Prediction:\n{response_text}\n")

        bleu = bleu_score(response_text, gold_answer)
        sem = semantic_score(response_text, gold_answer)
        f1 = f1_score(response_text, gold_answer)

        total_bleu += bleu
        total_semantic += sem
        total_f1 += f1

        llm_output.append({
            "index": i+1,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": response_text,
            "f1_score": f1,
            "semantic_score": sem,
            "return_doc_ids": return_doc_ids,
        })
 
    # Averaging Summary Information
    num_samples = len(qa_samples)
    print(f"Total samples: {num_samples}, average BLEU: {total_bleu / num_samples:.4f}, average F1: {total_f1 / num_samples:.4f}, average Semantic: {total_semantic / num_samples:.4f}, average precision: {total_precison / num_samples:.4f}, average reall: {total_recall / num_samples:.4f}")


    output_dir = f"output/{args.data}/{args.model}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build file paths for responses and their aggregated scores (include seed to avoid overwriting)
    response_file_path = f"{output_dir}/llm_output_{args.retrieval_method}_human_{args.k}_seed{args.seed}_response.json"
    scores_file_path = f"{output_dir}/llm_output_{args.retrieval_method}_human_{args.k}_seed{args.seed}_scores.json"

    if args.save == "true":
        # Persist the seed along with each record for full reproducibility
        for record in llm_output:
            record["seed"] = args.seed

        # Save detailed per-sample responses
        with open(response_file_path, "w", encoding="utf-8") as f:
            json.dump(llm_output, f, indent=2, ensure_ascii=False)

        # Save aggregated evaluation metrics only
        summary_metrics = {
            "total_samples": num_samples,
            "average_bleu": total_bleu / num_samples,
            "average_f1": total_f1 / num_samples,
            "average_semantic": total_semantic / num_samples,
            "average_precision": total_precison / num_samples,
            "average_recall": total_recall / num_samples,
            "seed": args.seed,
        }

        with open(scores_file_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2, ensure_ascii=False)

        print(f"Saved responses to {response_file_path} and scores to {scores_file_path}!")


if __name__ == "__main__":
    main()

