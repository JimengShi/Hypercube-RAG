SEED=42
echo "Running with seed ${SEED}..."
python qa_rag_hurricane_parallel_human.py \
  --data hurricane \
  --model gpt-4o \
  --retrieval_method hypercube \
  --k 3 \
  --seed ${SEED}