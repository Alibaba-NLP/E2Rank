import json
import random
import os
import transformers
import torch
from collections import defaultdict
from typing import Dict, Sequence
from torch.utils.data import Dataset


MIXED_RANK_DATA_INSTRUCTIONS = {
    "msmarco": (
        "Given a web search query, retrieval the documents that answer the query",
        "Given a web search query and some relevant documents, rerank the documents that answer the query"
    ),
    "nq": (
        "Given a question, retrieval Wikipedia documents that answer the question"
        "Given a question and some relevant documents, rerank the documents that answer the question"
    ),
    "hotpotqa": (
        "Given a multi-hop question, retrieval the documents that can help answer the question",
        "Given a multi-hop question and some relevant documents, rerank the documents that answer the question"
    ),
    "trivia": (
        "Retrieval Wikipedia documents that answer the question",
        "Given a question and some relevant Wikipedia documents, rerank the documents that answer the question"
    ),
    "t2ranking": (
        "Given a Chinese search query, retrieval the documents that answer the query",
        "Given a Chinese search query and some relevant documents, rerank the documents that answer the query"
    ),
    "dureader": (
        "Given a Chinese search query,r etrieval the documents that answer the query",
        "Given a Chinese search query and some relevant documents, rerank the documents that answer the query"
    ),
    "mmarco_chinese": (
        "Given a Chinese web search query, retrieval the documents that answer the query",
        "Given a Chinese web search query and some relevant documents, rerank the documents that answer the query"
    ),
    "cMedQAv2": (
        "Given a Chinese medical question, retrieval the documents that answer the question",
        "Given a Chinese medical question and some relevant documents, rerank the documents that answer the question"
    ),
    "colliee": (
        "Given a legal case, retrieval the relevant legal articles that can help analyze the case",
        "Given a legal case and some relevant legal articles, rerank the legal articles that can help analyze the case"
    ),
    "law_gpt": (
        "Given a Chinese legal case, retrieval the relevant legal articles that can help analyze the case",
        "Given a Chinese legal case and some relevant legal articles, rerank the legal articles that can help analyze the case"
    ),
    "miracl": (
        "Given a question, retrieval Wikipedia documents that answer the question",
        "Given a question and some relevant Wikipedia documents, rerank the documents that answer the question"
    ),
}

DEFAULT_INSTRUCTION = (
    "Given a query, retrieval the documents that are relevant to the query",
    "Given a query and some relevant documents, rerank the documents that are the most relevant to the query"
)


class MixedRankDataset(Dataset):
    rank_prompt_template = """{instruction}:
Documents:
{documents}
Query: {query}"""
    query_instuction_template = "Instruct: {task_description}\nQuery:{query}"

    def __init__(
        self,
        data_args: Dict,
        batch_size: int = 32,
        per_dataset_max_samples: int = 10000,
    ):
        self.data_args = data_args
        self.effective_batch_size = batch_size
        self.per_dataset_max_samples = per_dataset_max_samples
        self.use_listwise = data_args.use_listwise

        self.data = []
        self.load_data(data_args.data_path)

    def __len__(self):
        return len(self.data)

    def format_query(self, task, query):
        instruction = MIXED_RANK_DATA_INSTRUCTIONS.get(
            task, DEFAULT_INSTRUCTION)[0]
        return self.query_instuction_template.format(
            task_description=instruction,
            query=query
        )

    def format_rank_prompt(self, task, query, docs):
        instruction = MIXED_RANK_DATA_INSTRUCTIONS.get(
            task, DEFAULT_INSTRUCTION)[1]
        return self.rank_prompt_template.format(
            instruction=instruction,
            documents="\n".join([f"[{i}] {doc}" for i, doc in enumerate(docs, start=1)]),
            query=query
        )

    def load_data(self, data_path: str = None):

        all_samples = []

        with open(data_path, "r") as f:
            dataset_samples = [json.loads(line) for line in f.readlines()]

        data_map = defaultdict(list)
        for i, sample in enumerate(dataset_samples):
            task = sample.get("source", "unknown")
            data_map[task].append(i)
            if self.use_listwise:
                pseudo_query = self.format_rank_prompt(task, sample["query"], sample["document"])
            else:
                pseudo_query = None

            all_samples.append({
                "query": self.format_query(task, sample["query"]),
                "document": sample["document"],
                "pseudo_query": pseudo_query,
                "ranking": sample["ranking"],
            })

        for task, samples in data_map.items():
            random.shuffle(samples)

        datasets = list(data_map.keys())

        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, min(len(dataset_samples), self.per_dataset_max_samples), self.effective_batch_size):
                batch = dataset_samples[i : i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    print(f"Skip 1 batch for dataset {dataset}.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        print(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        return self.data[index]


class MixedRankDataCollator:

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token:
            if getattr(self.tokenizer, "eot_token", None):
                self.tokenizer.pad_token = self.tokenizer.eot_token
            else:
                self.tokenizer.pad_token = self.tokenizer.bos_token
        print(f"use ``{self.tokenizer.pad_token}`` as pad token for llm")

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        query_max_length = 512
        doc_max_length = 1024

        queries = [instance["query"] + self.tokenizer.pad_token for instance in instances]
        queries_encodings = self.tokenizer(
            queries,
            padding=True, truncation=True,
            max_length=query_max_length,
            return_tensors="pt",
        )

        docs = sum([instance["document"] for instance in instances], [])
        docs = [f"{doc}" + self.tokenizer.pad_token for doc in docs]
        docs_encodings = self.tokenizer(
            docs,
            padding=True, truncation=True,
            max_length=doc_max_length,
            return_tensors="pt",
        )

        if instances[0]["pseudo_query"] is not None:
            pseudo_queries = [instance["pseudo_query"] + self.tokenizer.pad_token for instance in instances]
            pseudo_query_encodings = self.tokenizer.apply_chat_template(
                [[{"role": "user", "content": prompt}] for prompt in pseudo_queries],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                padding=True, truncation=True, max_length=32768,
                return_tensors="pt", return_dict=True
            )
        else:
            pseudo_query_encodings = None

        if "ranking" in instances[0]:
            ranking = torch.tensor([instance["ranking"] for instance in instances]) - 1  # Convert to zero-based indexing
        else:
            ranking = None

        return {
            "query": queries_encodings,
            "document": docs_encodings,
            "pseudo_query": pseudo_query_encodings,
            "ranking": ranking
        }
