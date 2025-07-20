# Hypercube-RAG
<div align="left">
<img src="https://github.com/JimengShi/Hypercube-RAG/blob/main/figures/hypercube_rag.jpg" alt="hypercuberag" width="1000"/> 
</div>

### 1. Organization

- `QA`: saves the question-answering pairs
- `corpus`: saves the original corpus
- `hypercube`: constructs and saves hypercube
- `baselines`: includes the reproduced methods where the official codes are not available. We run graph-based models included in our work using their official GitHub repository. 
- `evaluation`: computes evaluation scores
- `utils`: helper functions
- `qa_norag.py`: script to run LLMs without retrieval
- `qa_rag_hurricane.py`: script to run Hypercube-RAG for hurricane dataset
- `qa_rag_geography.py`: script to run Hypercube-RAG for geography dataset
- `qa_rag_dam.py`: script to run Hypercube-RAG for dam dataset



### 2. Environment
```
conda create --name hypercube python==3.10
conda activate hypercube

pip install -r requirements.txt
```

### 3. Run the framework
#### 3.0 Construct knowledge hypercube 

- (**Optional**) You can skip this step if you want to run on our datasets since we have constructed the hypercube and saved the results in `hypercube` folder.

- If you want to use your own dataset, please first put the text corpus in `corpus` folder: `corpus/YOUR_DATA_FOLDER/YOUR_DATA_FILE`, then construct hypercube using a name entity recogition file `ner.py` file.

```
python corpus/YOUR_DATA_FOLDER/YOUR_ner.py
```


#### 3.1 Run the direct inference without RAG baseline
```
export CUDA_VISIBLE_DEVICES=GPU_ID
export OPENAI_API_KEY="your OPENAI_API_KEY"
```
In our work, we multiple datasets `hurricane`, `geography`, `aging_dam` with multiple LLMs, such as `gpt-4`, `gpt-4o`, `gpt-3.5-turbo`, `llama3`, `deepseek`, `qwen`. If you want to save the QA results, set the following "save" to true; otherwise, false.
```
python qa_norag.py --data `DATASET` --model `MODEL_NAME` --save `true`
```


#### 3.2 Run the Hypercube-RAG method
The following script takes the hurricane dataset and GPT-4o as an LLM base.

```
python qa_rag_hurricane.py --data hurricane --model gpt-4o --retrieval_method hypercube --save true
```



### 4. Evaluation
We have automatic metrics from the NLP domain and LLM-as-judge to evaluate the quality of answers.

#### 4.1 NLP metrics
```
python evaluation/nlp_metric.py --data hurricane --model gpt-4o --retrieval_method hypercube --metric all
```



#### 4.2 LLM-as-judge

```
python evaluation/llm_as_judge.py --data hurricane --model gpt-4o --retrieval_method hypercube
```

