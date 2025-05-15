# Hypercube-RAG

### 1. Organization
```bash
├── QA
│   ├── aging_dam
│   │   └── synthetic_qa.json
│   ├── geography
│   │   └── synthetic_qa.json
│   └── hurricane
│       └── synthetic_qa.json
├── README.md
├── baselines
│   ├── qa_rag_bm25.py
│   ├── qa_rag_contriever_dam.py
│   ├── qa_rag_contriever_hurricane.py
│   └── qa_rag_e5_hurricane.py
├── corpus
│   ├── aging_dam
│   │   └── corpus.txt
│   ├── geography
│   │   └── SciDCC-Geography.txt
│   └── hurricane
│       └── SciDCC-Hurricane.txt
├── evaluation
│   ├── llm_as_judge.py
│   └── nlp_metric.py
├── hypercube
│   ├── aging_dam
│   │   ├── dam.txt
│   │   ├── date.txt
│   │   ├── event.txt
│   │   ├── facility.txt
│   │   ├── location.txt
│   │   ├── ner_dam.py
│   │   ├── organization.txt
│   │   ├── person.txt
│   │   ├── quant.txt
│   │   └── watershed.txt
│   ├── geography
│   │   ├── date.txt
│   │   ├── event.txt
│   │   ├── location.txt
│   │   ├── ner_geo.py
│   │   ├── organization.txt
│   │   ├── person.txt
│   │   ├── run_rag.py
│   │   └── theme.txt
│   └── hurricane
│       ├── date.txt
│       ├── event.txt
│       ├── location.txt
│       ├── ner_hurricane.py
│       ├── organization.txt
│       ├── person.txt
│       └── theme.txt
├── qa_norag.py
├── qa_rag_dam.py
├── qa_rag_geography.py
├── qa_rag_hurricane.py
└── utils
    ├── metric.py
    └── topk_most_freq.py
```

### 2. Environment
```
conda create --name hypercube python==3.10
conda activate hypercube

pip install accelerate openai together geopy nltk rouge_score sentence_transformers spacy
```

### 3. Run the framework
#### Run the direct inference without RAG baseline
```
export CUDA_VISIBLE_DEVICES=GPU_ID
export OPENAI_API_KEY="your OPENAI_API_KEY"
```
In our work, we multiple datasets `hurricane`, `geography`, `aging_dam` with multiple LLMs, such as `gpt-4`, `gpt-4o`, `gpt-3.5-turbo`, `llama3`, `deepseek`, `qwen`. If you want to save the QA results, set the following "save" to true; otherwise, false.
```
python qa_norag.py --data `DATASET` --model `MODEL_NAME` --save `true`
```


#### Run the Hypercube-RAG method
The following script takes the hurricane dataset and GPT-4o as an LLM base.

```
python qa_rag_hurricane.py --data hurricane --model gpt-4o --retrieval_method hypercube --save true
```



### 4. Evaluation
We have automatic metrics from the NLP domain and LLM-as-judge to evaluate the quality of answers.

#### NLP metrics
```
python evaluation/nlp_metric.py --data hurricane --model gpt-4o --retrieval_method hypercube --metric all
```



#### LLM-as-judge

```
python evaluation/llm_as_judge.py --data hurricane --model gpt-4o --retrieval_method hypercube
```

