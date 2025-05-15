import os
import json
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import transformers
import torch
import numpy as np
import heapq
from openai import OpenAI
import pickle
from geopy.geocoders import GoogleV3

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

# read corpus
print('Reading the corpus file ...')
with open('raw_data/hurricane/SciDCC-Hurricane.txt') as f:
    readin = f.readlines()
    corpus = [line.strip() for line in tqdm(readin)]


# build embedding index
model = SentenceTransformer('intfloat/e5-base-v2')
doc_embeddings = model.encode(['passage: ' + text for text in corpus], normalize_embeddings=True)


# read and construct hypercube
print('Reading the hypercube files ...')

dimensions = ['date', 'event', 'location', 'organization', 'person', 'theme']
hypercube = {'date': defaultdict(list),
             'event': defaultdict(list),
             'location': defaultdict(list),
             'organization': defaultdict(list),
             'person': defaultdict(list),
             'theme': defaultdict(list),}


for dimension in dimensions:
    with open(f'hurricane/{dimension}.txt') as f:
        readin = f.readlines()
        for i, line in tqdm(enumerate(readin), total=len(readin), desc=f"{dimension}"):
            tmp = json.loads(line)
            for k in tmp:
                hypercube[dimension][k].append(i)
                
                embed_string(target_str=k, emb_model=model, dict_path='./ent2emb.pkl', )


def get_docs_from_cells(cells):
    if cells is None: return []
    tmp_ids = []
    doc_ids = set(list(range(len(corpus))))
    for k, v in cells.items():
        assert k in hypercube
        
        for vv in v:
            if vv in hypercube[k]:
                tmp_ids.extend(hypercube[k][vv])
            else:
                vv_emb = ent2emb[vv]
                for cand in hypercube[k]:
                    if ent2emb[cand] @ vv_emb > 0.9: 
                        tmp_ids.extend(hypercube[k][cand])
                        
    doc_ids = doc_ids.intersection(set(tmp_ids))
    # Todo: whether limit the number of documents
    return list(doc_ids)


def main(llm, query, cells, k=3, retrieval_method='union'):
    assert llm in ['gpt-4o', 'llama3.1']
    assert retrieval_method in ['hypercube', 'semantic', 'union']

    # set up instruction
    instruction = 'Answer the query based on the given retrieved documents. Documents:\n'

    # get docs from cells
    structure_docs = get_docs_from_cells(cells)

    # get query embedding
    query_embedding = model.encode(['query: ' + query])
    scores = np.matmul(doc_embeddings, query_embedding.transpose())[:,0]
    print(">>>>> scores:", scores)
    semantic_docs = [index for _, index in heapq.nlargest(k, ((v, i) for i, v in enumerate(scores)))]

    # print(f'Doc ids by semantic retrieval: {semantic_docs}')
    # print(f'Doc ids by hypercube retrieval: {structure_docs}')

    ### (To do) we can add any of searching combination
    if retrieval_method == 'hypercube':
        doc_ids = structure_docs
        print(f'Doc ids by hypercube retrieval: {[id + 1 for id in doc_ids]}')
    elif retrieval_method == 'semantic':
        doc_ids = semantic_docs
        print(f'Doc ids by semantic retrieval: {[id + 1 for id in doc_ids]}')
    elif retrieval_method == 'union':
        doc_ids = list(set(structure_docs).union(set(semantic_docs)))
        print(f'Doc ids by union retrieval: {[id + 1 for id in doc_ids]}')       
    ### (End to do)
    
    docs = '\n\n'.join([f"Document {idx + 1}: {corpus[doc_id]}" for idx, doc_id in enumerate(doc_ids)])
    # print(docs)

    # set up the LLM
    if llm == 'llama3.1':
        pipeline = transformers.pipeline(
                "text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
            )
        messages = [
            {"role": "system", "content": instruction + docs},
            {"role": "user", "content": 'Query: ' + query + '\nAnswer:'},
            ]
        res = pipeline(messages, max_new_tokens=256)
        res = res[0]['generated_text'][-1]['content']
    elif llm == 'gpt-4o':
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

    print(f'########## Model {llm} Answer ##########')
    print(res)


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
        f"Question: How Indian Monsoons Influence Hurricane Paths?\n"
        f"Example Output:\n"
        f"Query 1:\n"
        f"query_dimension: 'location'; query_content: 'Indian Monsoons';\n"
        f"Query 2:\n"
        f"query_dimension: 'hurriance'; query_content: 'Hurricane Paths' (negative example: 'hurriance', not self-complete and informative);\n"
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
    

if __name__ == '__main__':
    llm = 'gpt-4o' # gpt-4o, llama3.1
    
    # query_list = [
    #             'What did Professor Amy Pruden say about Hurricane Maria impacts?' 
    #             # 'How Indian Monsoons Influence Hurricane Paths?'  
    #             # 'How Indian Monsoons influence Atlantic hurricane paths?'
    #             # 'How can ENSO and La Niña impact Atlantic hurricane season?'
    #               ]
    query_list = load_dataset('QA/hurricane/qa_factoid.json')
    # query_list = load_dataset(data_path)

    for sample in query_list[:2]:
        query = sample['question']

        print(f"--------\nQuery: {query}\n--------\n")

        cells = decompose_query(query=query)
        print(f"Identified cells from the query: {cells} \n")


        print(f"---------hypercube-RAG---------")
        main(llm, query, cells, retrieval_method='hypercube')
        print("\n")
        
        print(f"---------semantic-RAG---------")
        main(llm, query, cells, retrieval_method='semantic')
        print("\n")

        # if 'location' in cells:
        #     new_loc = []
        #     for loc in cells['location']:
        #         geolocator = GoogleV3(api_key = os.environ["GOOGLE_API_KEY"])
        #         location = geolocator.geocode(loc)
        #         if location is None:
        #             continue
        #         location_address = location.raw['address_components']
        #         for item in location_address:
        #             new_loc.append(item['long_name'])
        #         # unique the location
        #         new_loc = list(set(new_loc))

        #     # print(f"new_loc: {new_loc}")
        #     # embed the location address
        #     [embed_string(target_str=new_loc_item, emb_model=model, dict_path='./ent2emb.pkl', ) for new_loc_item in new_loc]

        #     # update cells['location'], union of new_loc and cells['location']
        #     cells['location'] = list(set(cells['location']).union(set(new_loc)))

        # print(f"---------hypercube+domain-RAG---------")
        # print(f"Identified cells from the query+domain: {cells}")
        # main(llm, query, cells, retrieval_method='hypercube')
