<div align="center">
<h1>E2Rank: Your Text Embedding can Also be an Effective and Efficient Listwise Reranker</h1>

<a href="https://Alibaba-NLP.github.io/E2Rank/">🤖 Website</a> | 
<a href="https://arxiv.org/pdf/2501.07572">📄 Arxiv Paper</a> | 
<a href="https://huggingface.co/collections/Alibaba-NLP/e2rank">🤗 Huggingface Collection</a> |
<a href="# 🚩Citation">🚩 Citation</a>

</div>

# 📌 Introduction

We introduce $\textrm{E}^2\text{Rank}$, 
meaning **E**fficient **E**mbedding-based **Rank**ing
(also meaning **Embedding-to-Rank**), 
which extends a single text embedding model
to perform both high-quality retrieval and listwise reranking,
thereby achieving strong effectiveness with remarkable efficiency. 

By applying cosine similarity between the query and
document embeddings as a unified ranking function, the listwise ranking prompt,
which is constructed from the original query and its candidate documents, serves
as an enhanced query enriched with signals from the top-K documents, akin to
pseudo-relevance feedback (PRF) in traditional retrieval models. This design 
preserves the efficiency and representational quality of the base embedding model
while significantly improving its reranking performance. 

Empirically, E2Rank achieves state-of-the-art results on the BEIR reranking benchmark 
and demonstrates competitive performance on the reasoning-intensive BRIGHT benchmark,
with very low reranking latency. We also show that the ranking training process
improves embedding performance on the MTEB benchmark. 
Our findings indicate that a single embedding model can effectively unify retrieval and reranking,
offering both computational efficiency and competitive ranking accuracy.

**Our work highlights the potential of single embedding models to serve as unified retrieval-reranking engines, offering a practical, efficient, and accurate alternative to complex multi-stage ranking systems.**


<div align="center">
    <img src="assets/cover.png" width="80%" height="auto" />
    <p style="width: 70%; margin-left: auto; margin-right: auto">
        <b>(a)</b> Overview of E2Rank. <b>(b)</b> Average reranking performance on the BEIR benchmark, E2Rank outperforms other baselines. <b>(c)</b> Reranking latency per query on the Covid dataset, E2Rank can achieve several times the acceleration compared with RankQwen3.
    </p>
</div>

# 🚀 Quick Start

## Model List

| Supported Task              | Model Name           | Size | Layers | Sequence Length | Embedding Dimension | Instruction Aware |
|-----------------------------|----------------------|------|--------|-----------------|---------------------|-------------------|
| **Embedding + Reranking**   | [Alibaba-NLP/E2Rank-0.6B](https://huggingface.co/Alibaba-NLP/E2Rank-0.6B) | 0.6B | 28     | 32K             | 1024                | Yes            |
| **Embedding + Reranking**   | [Alibaba-NLP/E2Rank-4B](https://huggingface.co/Alibaba-NLP/E2Rank-4B)     | 4B   | 36     | 32K             | 2560                | Yes            |
| **Embedding + Reranking**   | [Alibaba-NLP/E2Rank-8B](https://huggingface.co/Alibaba-NLP/E2Rank-8B)     | 8B   | 36     | 32K             | 4096                | Yes            |
| Embedding Only              | [Alibaba-NLP/E2Rank-0.6B-Embedding-Only](https://huggingface.co/Alibaba-NLP/E2Rank-0.6B-Embedding-Only) | 0.6B | 28     | 32K             | 1024                | Yes         |
| Embedding Only              | [Alibaba-NLP/E2Rank-0.6B-Embedding-Only](https://huggingface.co/Alibaba-NLP/E2Rank-4B-Embedding-Only)   | 4B   | 36     | 32K             | 2560                | Yes         |
| Embedding Only              | [Alibaba-NLP/E2Rank-0.6B-Embedding-Only](https://huggingface.co/Alibaba-NLP/E2Rank-8B-Embedding-Only)   | 8B   | 36     | 32K             | 4096                | Yes         |


> **Note**:
> - `Embedding Only` indicates that the model is trained only with the constrative learning and support embedding tasks, while `Embedding + Reranking` indicates the **full E2Rank model** trained with both embedding and reranking objectives (for more detals, please refer to the [paper]()). 
> - `Instruction Aware` notes whether the model supports customizing the input instruction according to different tasks.

## Usage

### Embedding Model

The usage of E2Rank as an embedding model is similar to [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding). The only difference is that Qwen3-Embedding will automatically append an EOS token, while E2Rank requires users to manully append the special token `<|endoftext|>` at the end of each input text.


**vLLM Usage (recommended)**

```python
# Requires vllm>=0.8.5
import torch
import vllm
from vllm import LLM
from vllm.config import PoolerConfig

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = [
    get_detailed_instruct(task, 'What is the capital of China?'),
    get_detailed_instruct(task, 'Explain gravity')
]
# No need to add instruction for retrieval documents
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]
input_texts = queries + documents
input_texts = [t + "<|endoftext|>" for t in input_texts]

model = LLM(
    model="Alibaba-NLP/E2Rank-0.6B",
    task="embed", 
    override_pooler_config=PoolerConfig(pooling_type="LAST", normalize=True)
)

outputs = model.embed(input_texts)
embeddings = torch.tensor([o.outputs.embedding for o in outputs])
scores = (embeddings[:2] @ embeddings[2:].T)
print(scores.tolist())
# [[0.5958386659622192, 0.030148349702358246], [0.060259245336055756, 0.5595865249633789]]
```

<details>
<summary><b>Transformers Usage</b></summary>

```python
# Requires transformers>=4.51.0
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = [
    get_detailed_instruct(task, 'What is the capital of China?'),
    get_detailed_instruct(task, 'Explain gravity')
]
# No need to add instruction for retrieval documents
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]
input_texts = queries + documents
input_texts = [t + "<|endoftext|>" for t in input_texts]

tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/E2Rank-0.6B', padding_side='left')
model = AutoModel.from_pretrained('Alibaba-NLP/E2Rank-0.6B')

max_length = 8192

# Tokenize the input texts
batch_dict = tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)
batch_dict.to(model.device)
with torch.no_grad():
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:2] @ embeddings[2:].T)

print(scores.tolist())
# [[0.5950675010681152, 0.030417663976550102], [0.061970409005880356, 0.562691330909729]]
```
</details>


### Reranking

For using E2Rank as a reranker, you only need to perform additional processing on the query by adding (part of) the docs that needs to be reranked to the *listwise prompt*, while the rest is the same as using the embedding model.

**vLLM Usage (recommended)**

```python
# Requires vllm>=0.8.5
import torch
import vllm
from vllm import LLM
from vllm.config import PoolerConfig

model = LLM(
    model="./checkpoints/E2Rank-0.6B",
    task="embed", 
    override_pooler_config=PoolerConfig(pooling_type="LAST", normalize=True)
)
tokenizer = model.get_tokenizer()

def get_listwise_prompt(task_description: str, query: str, documents: list[str], num_input_docs: int = 20) -> str:
    input_docs = documents[:num_input_docs]
    input_docs = "\n".join([f"[{i}] {doc}" for i, doc in enumerate(input_docs, start=1)])
    messages = [{
        "role": "user", 
        "content": f'{task_description}\nDocuments:\n{input_docs}Search Query:{query}'
    }]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text

task = 'Given a web search query and some relevant documents, rerank the documents that answer the query:'

queries = [
    'What is the capital of China?',
    'Explain gravity'
]

# No need to add instruction for retrieval documents
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]
documents = [doc + "<|endoftext|>" for doc in documents]

pseudo_queries = [
    get_listwise_prompt(task, queries[0], documents),
    get_listwise_prompt(task, queries[1], documents)
]  # no need to add the EOS token here

input_texts = pseudo_queries + documents

outputs = model.embed(input_texts)
embeddings = torch.tensor([o.outputs.embedding for o in outputs])
scores = (embeddings[:2] @ embeddings[2:].T)
print(scores.tolist())
# [[0.8516960144042969, 0.24043934047222137], [0.33099934458732605, 0.7905282974243164]]
```

<details>
<summary><b>Transformers Usage</b></summary>

```python
# Requires transformers>=4.51.0
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained('./checkpoints/E2Rank-0.6B', padding_side='left')
model = AutoModel.from_pretrained('./checkpoints/E2Rank-0.6B')


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_listwise_prompt(task_description: str, query: str, documents: list[str], num_input_docs: int = 20) -> str:
    input_docs = documents[:num_input_docs]
    input_docs = "\n".join([f"[{i}] {doc}" for i, doc in enumerate(input_docs, start=1)])
    messages = [{
        "role": "user", 
        "content": f'{task_description}\nDocuments:\n{input_docs}Search Query:{query}'
    }]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text

task = 'Given a web search query and some relevant documents, rerank the documents that answer the query:'

queries = [
    'What is the capital of China?',
    'Explain gravity'
]

# No need to add instruction for retrieval documents
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]
documents = [doc + "<|endoftext|>" for doc in documents]

pseudo_queries = [
    get_listwise_prompt(task, queries[0], documents),
    get_listwise_prompt(task, queries[1], documents)
]  # no need to add the EOS token here

input_texts = pseudo_queries + documents


max_length = 8192
# Tokenize the input texts
batch_dict = tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)
batch_dict.to(model.device)
with torch.no_grad():
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:2] @ embeddings[2:].T)

print(scores.tolist())
# [[0.8513513207435608, 0.24268491566181183], [0.33154672384262085, 0.7923378944396973]]
```
</details>

### End-to-end search

Since E2Rank extends a single text embedding model to perform both high-quality retrieval and listwise reranking, you can directly use it to build an end-to-end search system. By reusing the embeddings computed during the retrieval stage, E2Rank only need to compute the pseudo query's embedding and can efficiently rerank the retrieved documents with minimal additional computational overhead.

Example code is coming soon.


# 🛠 Reproducibility

We provide detailed instructions in this section for reproducing the training and evaluation results of our models in the paper.

## Dependencies

```bash
conda create -n e2rank python=3.10
git clone https://github.com/Alibaba-NLP/E2Rank.git
cd E2Rank

# Install requirements
pip install -r requirements.txt

# for evaluation, we also use LLM4Ranking framework
pip install git@github.com:liuqi6777/llm4ranking.git
```

## Training

We provide the training scripts for the 2nd stage training (training from the existing embedding model), and the checkpoints of embedding-only model used in the paper are released on [Huggingface](https://huggingface.co/collections/Alibaba-NLP/e2rank). You can directly download and use them for the next stage of training the full E2Rank model. If you want to re-train the embedding-only model, please refer to the paper for more details.

**Download the Datasets**

You can download the pre-processed and labeled datasets from [here](https://huggingface.co/datasets/Alibaba-NLP/E2Rank_ranking_datasets) and place them in the `data/` directory:

```bash
mkdir data
hf download Alibaba-NLP/E2Rank_ranking_datasets train.jsonl --local-dir ./data/ --repo-type dataset 
```

For more details about the datasets, please refer to the original paper.

**Train E2Rank-0.6B**

To train the E2Rank-0.6B model, run the following script (see `scripts/train_e2rank_0.6b.sh` for more details):

```bash
bash ./scripts/train_e2rank_0.6b.sh
```

For training E2Rank-4B and E2Rank-8B models, the training process is similar. The checkpoints will be saved in the `checkpoints/` directory.


## Evaluation

### Reranking Benchmarks (TREC DL, BEIR and BRIGHT)

The implementation of evaluation on reranking benchmarks are based on [LLM4Ranking](https://github.com/liuqi6777/llm4ranking) framework. To evaluate the model's reranking performance on BEIR and BRIGHT benchmarks, run the following script:

```bash
export VLLM_USE_MODELSCOPE=False

model_name="Alibaba-NLP/E2Rank-0.6B"
datasets="dl19 dl20"
retriever="bm25"

python src/eval.py \
    --model ${model_name} \
    --rank-method listwise \
    --datasets ${datasets} \
    --retriever ${retriever} \
    --topk 100 \
    --save-to "./results/rerank/all_results.jsonl"
```

- `model_name`: Path or name of the model weights file (e.g., "Alibaba-NLP/E2Rank-0.6B").
- `datasets`: A list of the names of datasets to be evaluated (e.g., `dl19 dl20 covid ...`). For full dataset names, please refer to `src/eval.py`.
- `retriever`: The retriever used to retrieve the initial candidate documents (Supports `bm25` (for all datasets), `reasonir` (for BRIGHT), et, al, see [https://huggingface.co/datasets/liuqi6777/retrieval_results](liuqi6777/retrieval_results) for all released first-stage retrieval results).
- `save-to`: The jsonl file to save the evaluation results. The trec-format running file will also be stored in the dictory of `outputs`.

> **Note**:
> - Due to the differences in the hardware (e.g., the type and the number of GPUs used) and software environments, the evaluation results may vary slightly from those reported in the paper, which is normal.


### MTEB

To evaluate the model on MTEB benchmark, run the following script:

```bash
bash eval_mteb/scripts/run_mteb.sh ${model_path} ${model_name} ${benchmark_name}
```

- `model_path`: Path or name of the model weights file (e.g., "Alibaba-NLP/E2Rank-0.6B").
- `model_name`: Name of the model, used for naming the result directory (e.g., "Alibaba-NLP/E2Rank-0.6B").
- `benchmark_name`: Name of the benchmark (e.g., "MTEB(eng, v1)" or "MTEB(eng, v2)").

Evaluation results will be saved in the directory: `results/mteb/`. Each task's results will be stored in a separate JSON file.

For summarizing experimental results, run the following script:

```bash
python3 eval_mteb/summary.py results/mteb/${model_name}/${model_name}/no_version_available ${benchmark_name}
```

The implementation of evaluation on MTEB are modified from [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/README.md). Sincere thanks for their efforts.

# 🚩 Citation

If this work is helpful, please kindly cite as:

```bibtext

```

If you have any questions, feel free to contact us via qiliu6777[AT]gmail.com or create an issue.

<!-- ## Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Alibaba-NLP/E2Rank&type=Date)](https://www.star-history.com/#Alibaba-NLP/WebAgent&Date)

</div> -->