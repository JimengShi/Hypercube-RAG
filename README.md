# Hypercube-RAG



### Environment
```
conda create --name hypercube python==3.10
conda activate hypercube

pip install accelerate openai together geopy nltk rouge_score sentence_transformers spacy
```

### Run the framework
#### Run the direct inference without RAG baseline
```
export OPENAI_API_KEY="your OPENAI_API_KEY"



export GOOGLE_API_KEY="AIzaSyDdtKzzFEURNX9GstezXI_j5kdCPUyYL7E"

CUDA_VISIBLE_DEVICES=2 python run.py
```
LLM backbone: You can select LLM backbones in line 101 and we currently support gpt-4o and llama3.1.

Query: Type it in line 102.

#### Run the hypercube RAG method
```
python run_rag.py
```

LLM backbone: You can select LLM backbones in line 101 and we currently support gpt-4o and llama3.1.

Query: Type it in line 102.

Hypercube cells: You can identify the hypercube cells in line 103. It should be a dictionary, where keys (str) correspond to the dimensions and values (list) correspond to the dimension values you are interested in. You can put multiple dimension keys in the dictionary and the hypercube search will return all the documents in the intersection of your provided dimension values.

Search: We currently support both hypercube-based search (```structure_docs``` in line 57) and text semantic search (```semantic_docs``` in line 62, you can adjust ```k``` which corresponds to how many docs you would like to return based on semantic search). You can design any strategy to combine the results from the two search methods around line 66 to get the final retrieved documents ```doc_ids```.


`python utils/reindex.py --data_path QA/hurricane/synthetic_qa.json`
`python utils/reindex.py --data_path QA/geography/synthetic_qa.json`
`python utils/reindex.py --data_path QA/aging_dam/synthetic_qa.json`


### QA
#### Hurricane
`python qa_norag.py --data hurricane --model gpt-4o --save true`
`python qa_rag_hurricane.py --data hurricane --model gpt-4o --retrieval_method hypercube --save true`
`python qa_rag_hurricane.py --data hurricane --model gpt-4o --retrieval_method semantic --save true`

#### Geography
`python qa_norag.py --data geography --model gpt-4o --save true`
`python qa_rag_geography.py --data geography --model gpt-4o --retrieval_method hypercube --save true`
`python qa_rag_geography.py --data geography --model gpt-4o --retrieval_method semantic --save true`


#### Aging Dam
`python qa_norag.py --data aging_dam --model gpt-4o --save true`
`python qa_rag_dam.py --data aging_dam --model gpt-4o --retrieval_method hypercube --save true`
`python qa_rag_dam.py --data aging_dam --model gpt-4o --retrieval_method semantic --save true`



### Evaluation

#### nlp metrics
`python evaluation/nlp_metric.py --data hurricane --model gpt-4o --retrieval_method none --metric all`
`python evaluation/nlp_metric.py --data hurricane --model gpt-4o --retrieval_method hypercube --metric all`
`python evaluation/nlp_metric.py --data hurricane --model gpt-4o --retrieval_method semantic --metric all`


`python evaluation/nlp_metric.py --data geography --model gpt-4o --retrieval_method none --metric all`
`python evaluation/nlp_metric.py --data geography --model gpt-4o --retrieval_method hypercube --metric all`
`python evaluation/nlp_metric.py --data geography --model gpt-4o --retrieval_method semantic --metric all`

`python evaluation/nlp_metric.py --data aging_dam --model gpt-4o --retrieval_method none --metric all`
`python evaluation/nlp_metric.py --data aging_dam --model gpt-4o --retrieval_method hypercube --metric all`
`python evaluation/nlp_metric.py --data aging_dam --model gpt-4o --retrieval_method semantic --metric all`



#### llm as judge
`python evaluation/llm_as_judge.py --data hurricane --model gpt-4o --retrieval_method none`
`python evaluation/llm_as_judge.py --data hurricane --model gpt-4o --retrieval_method hypercube`
`python evaluation/llm_as_judge.py --data hurricane --model gpt-4o --retrieval_method semantic`


`python evaluation/llm_as_judge.py --data geography --model gpt-4o --retrieval_method none`
`python evaluation/llm_as_judge.py --data geography --model gpt-4o --retrieval_method hypercube`
`python evaluation/llm_as_judge.py --data geography --model gpt-4o --retrieval_method semantic`


`python evaluation/llm_as_judge.py --data aging_dam --model gpt-4o --retrieval_method none`
`python evaluation/llm_as_judge.py --data aging_dam --model gpt-4o --retrieval_method hypercube`
`python evaluation/llm_as_judge.py --data aging_dam --model gpt-4o --retrieval_method semantic`
