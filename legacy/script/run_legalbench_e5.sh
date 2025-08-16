for SEED in {42..44}; do
  echo "Running with seed ${SEED}..."
  python qa_rag_legalbench.py \
    --data legalbench \
    --model gpt-4o \
    --retrieval_method semantic \
    --k 5 \
    --seed ${SEED}
done