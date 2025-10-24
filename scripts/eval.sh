#!/bin/bash

set -e

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MODELSCOPE=False

model_name=Alibaba-NLP/E2Rank-0.6B

# trec dl + beir
python src/eval.py \
    --model $model_name \
    --rank-method listwise \
    --datasets dl19 dl20 covid nfc touche dbpedia scifact signal news robust04 \
    --retriever bm25 \
    --topk 100 \
    --save-to results/rerank/all_results.jsonl

# bright
python src/eval.py \
    --model $model_name \
    --rank-method listwise \
    --datasets bright-biology bright-earth-science bright-economics bright-psychology bright-robotics bright-stackoverflow bright-sustainable-living bright-pony bright-leetcode bright-aops bright-theoremqa-theorems bright-theoremqa-questions \
    --retriever bm25 \
    --topk 100 \
    --save-to results/rerank/all_results.jsonl

echo "All evaluations completed."

