import argparse
import json
import os
import tempfile
import time
import random
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams, PoolingParams
from vllm.config import PoolerConfig

from functools import partial
from llm4ranking.ranker import ListwiseSilidingWindowReranker
from llm4ranking.evaluation.trec_eval import trec_eval


EVAL_PROMPTS = {
    "dl19": (
        "Given a web search query and some relevant documents, rerank the documents that answer the query:\nDocuments:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}"
    ),
    "dl20": (
        "Given a web search query and some relevant documents, rerank the documents that answer the query:\nDocuments:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}"
    ),
    "covid": (
        "Given a query on COVID-19 and some relevant documents, rerank the documents that answer the query:\nDocuments:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a query on COVID-19, retrieve documents that answer the query\nQuery:{query}"
    ),
    "nfc": (
        "Given a question and some relevant documents, rerank the documents that best answer the query:\nDocuments:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery:{query}"
    ),
    "touche": (
        "Given a question and some arguments, rerank the detailed and persuasive arguments that answer the question:\nArguments:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a question, retrieve detailed and persuasive arguments that answer the question\nQuery:{query}"
    ),
    "dbpedia": (
        "Given a query and some relevant entities, rerank the best descriptions from DBPedia:\nEntities:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a query, retrieve relevant entity descriptions from DBPedia\nQuery:{query}"
    ),
    "scifact": (
        "Given a scientific claim and some relevant documents, rerank the documents that best support or refute the claim:\nDocuments:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery:{query}"
    ),
    "signal": (
        "Given a news article title and some relevant tweets, rerank the tweets that are most relevant to the query:\nTweets:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a news article title, retrieve relevant tweets\nQuery:{query}"
    ),
    "news": (
        "Given a news article title and some relevant documents, rerank the documents that best answer the query:\nDocuments:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a news headline, retrieve relevant relevant news articles that provide important context or background information\nQuery:{query}"
    ),
    "robust04": (
        "Given a web search query and some relevant documents, rerank the documents that answer the query:\nDocuments:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a newswire query, retrieve relevant news articles that address the query.\nQuery:{query}"
    ),

    "bright-aops": (
        "Given a Math problem and some relevant examples, rerank the examples that help answer the problem:\nExamples:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a Math problem, retrieve relevant examples that help answer the problem\nQuery:{query}"
    ),
    "bright-biology": (
        "Given a biology post and some relevant passages, rerank the passages that help answer the post:\nPassages:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a post, retrieve relevant passages that help answer the post\nQuery:{query}"
    ),
    "bright-earth_science": (
        "Given an earth science post and some relevant passages, rerank the passages that help answer the post:\nPassages:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a post, retrieve relevant passages that help answer the post\nQuery:{query}"
    ),
    "bright-economics": (
        "Given an economics post and some relevant passages, rerank the passages that help answer the post:\nPassages:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a economics post, retrieve relevant passages that help answer the post\nQuery:{query}"
    ),
    "bright-leetcode": (
        "Given a coding problem and some relevant examples, rerank the examples that help answer the problem:\nExamples:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a coding problem, retrieve relevant examples that help answer the problem\nQuery:{query}"
    ),
    "bright-pony": (
        "Given a question about the Pony programming language and some relevant passages, rerank the passages that help answer the question:\nPassages:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a question about pony program language, retrieve relevant passages that help answer the question\nQuery:{query}"
    ),
    "bright-psychology": (
        "Given a psychology post and some relevant passages, rerank the passages that help answer the post:\nPassages:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a psychology post, retrieve relevant passages that help answer the post\nQuery:{query}"
    ),
    "bright-theoremqa_questions": (
        "Given a Math problem and some relevant examples, rerank the examples that help answer the problem:\nExamples:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a Math problem, retrieve relevant examples that help answer the problem\nQuery:{query}"
    ),
    "bright-theoremqa_theorems": (
        "Given a Math problem and some relevant theorems, rerank the theorems that help answer the problem:\nTheorems:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a Math problem, retrieve relevant theorems that help answer the problem\nQuery:{query}"
    ),
    "bright-robotics": (
        "Given a robotics post and some relevant passages, rerank the passages that help answer the post:\nPassages:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a robotics post, retrieve relevant passages that help answer the post\nQuery:{query}"
    ),
    "bright-stackoverflow": (
        "Given a StackOverflow post and some relevant passages, rerank the passages that help answer the post:\nPassages:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a stackoverflow post, retrieve relevant passages that help answer the post\nQuery:{query}"
    ),
    "bright-sustainable_living": (
        "Given a sustainable living post and some relevant passages, rerank the passages that help answer the post:\nPassages:\n{documents}\nSearch Query: {query}",
        "Instruct: Given a sustainable_living post, retrieve relevant passages that help answer the post\nQuery:{query}"
    )
}

encode_doc_times = []
encode_query_times = []

class EvalModel:

    listwise_prompt_template = "Given a web search query and some relevant documents, rerank the documents that answer the query:\nDocuments:\n{documents}\nSearch Query: {query}"
    query_prompt_template = "Instruct: Given a web search query and a document, determine how relevant the document is to the query.\nQuery:{query}"

    def __init__(self, model: LLM, rank_method: str = "listwise", num_input_docs: int = 20):
        self.model = model
        self.tokenizer = self.model.get_tokenizer()
        self.rank_method = rank_method
        self.num_input_docs = num_input_docs

    def set_prompt_template(self, dataset: str):
        if dataset in EVAL_PROMPTS:
            self.listwise_prompt_template = EVAL_PROMPTS[dataset][0]
            self.query_prompt_template = EVAL_PROMPTS[dataset][1]

    def __call__(self, query: str, docs: list[str]) -> list[int]:

        t = time.time()
        doc_max_len = 1024
        docs = [" ".join(x.split()[:doc_max_len]) for x in docs]
        input_docs = docs[:self.num_input_docs]
        text = self.listwise_prompt_template.format(
            query=query,
            documents="\n".join([f"[{i}] {doc}" for i, doc in enumerate(input_docs, start=1)]),
        )
        docs = [doc + "<|endoftext|>" for doc in docs]

        outputs = self.model.embed(
            docs,
            use_tqdm=False,
        )
        d_reps = []
        for output in outputs:
            d_reps.append(output.outputs.embedding)
        d_reps = torch.tensor(d_reps, dtype=torch.float32)
        encode_doc_times.append(time.time() - t)

        t = time.time()
        messages = [{"role": "user", "content": text}]
        if self.rank_method == "listwise":
            query = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # query = query + "<|endoftext|>"
        else:
            query = self.query_prompt_template.format(query=query) + "<|endoftext|>"
        q_reps = self.model.embed(
            query,
            use_tqdm=False
        )[0].outputs.embedding
        q_reps = torch.tensor(q_reps, dtype=torch.float32).unsqueeze(0)

        scores = (q_reps @ d_reps.T).flatten()
        ranking = scores.argsort(descending=True).tolist()
        encode_query_times.append(time.time() - t)

        return ranking


def simple_evaluate(
    model: EvalModel,
    datasets: list[str],
    retriever: str = "bm25",
    topk: int = 100,
    max_samples: int = None,
    use_sliding_window: bool = True,
    num_passes: int = 1,
    order: str = "initial",
    output_dir: str = None,
):

    ranker = ListwiseSilidingWindowReranker()

    results = {}
    results["output_dir"] = output_dir

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        try:
            print(f"Evaluating dataset {dataset}...")
            if model.rank_method == "embedding":
                model.set_prompt_template(dataset)
            rerank = partial(
                ranker.rerank,
                ranking_func=model,
                window_size=20 if use_sliding_window else topk,
                step=10,
            )

            data = load_dataset("liuqi6777/retrieval_results", data_files=f"{retriever}/{dataset}_top{topk}.jsonl", split="train").to_list()

            results[dataset] = {}
            if max_samples is not None:
                data = data[:max_samples]

            if dataset.startswith("bright"):
                task_name = dataset.removeprefix("bright-").replace("-", "_")
                examples = load_dataset('xlangai/bright', 'examples')[task_name]
                excluded_ids = {}
                for e in examples:
                    excluded_ids[e['id']] = e['excluded_ids']
            else:
                excluded_ids = None

            print(f"First stage metrics:")
            with tempfile.NamedTemporaryFile("w") as f:
                write_results(data, f)
                f.flush()
                metrics = trec_eval(dataset, f.name, excluded_ids)

            prev_results = data
            for pass_ in range(num_passes):
                rerank_results = []
                for i in tqdm(range(len(prev_results))):
                    if pass_ == 0:
                        if order == "reverse":
                            prev_results[i]["hits"].reverse()
                        elif order == "random":
                            random.shuffle(prev_results[i]["hits"])
                    _, rerank_indices, *_ = rerank(
                        query=prev_results[i]["query"],
                        candidates=[x["content"] for x in prev_results[i]["hits"]],
                        return_record=False,
                        return_indices=True
                    )
                    rerank_results.append({
                        "query": prev_results[i]["query"],
                        "hits": [prev_results[i]["hits"][j] for j in rerank_indices]
                    })
                prev_results = rerank_results

                if output_dir is not None:
                    output_file = os.path.join(
                        output_dir,
                        f"eval_{dataset}_top{topk}_pass{pass_}.txt"
                    )
                    with open(output_file, "w") as f:
                        write_results(rerank_results, f)
                    metrics = trec_eval(dataset, output_file, excluded_ids)
                else:
                    with tempfile.NamedTemporaryFile("w") as f:
                        write_results(rerank_results, f)
                        f.flush()
                        metrics = trec_eval(dataset, f.name, excluded_ids)

                results[dataset]["pass" + str(pass_)] = {}
                results[dataset]["pass" + str(pass_)]["metrics"] = metrics
        except Exception as e:
            print(f"Error evaluating dataset {dataset}: {e}")

    return results


def write_results(rerank_results, file_obj):
    for i, item in enumerate(rerank_results):
        hits = item["hits"]
        for j, hit in enumerate(hits):
            file_obj.write(f"{hit['qid']} Q{i} {hit['docid'].replace(' ', '_')} {j + 1} {(100 - j)} rank")
            file_obj.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--rank-method", type=str, default="listwise", choices=["listwise", "embedding"])
    parser.add_argument("--num-input-docs", type=int, default=20)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sliding-window", default=False, action="store_true")
    parser.add_argument("--num-passes", type=int, default=1)
    parser.add_argument("--order", type=str, default="initial", choices=["initial", "reverse", "random"])
    parser.add_argument("--save-to", type=str, default=None)
    args = parser.parse_args()
    print(args)

    random.seed(42)

    model = LLM(
        args.model,
        task="embed",
        override_pooler_config=PoolerConfig(pooling_type="LAST", normalize=True),
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
    )
    model = EvalModel(
        model,
        rank_method=args.rank_method,
        num_input_docs=args.num_input_docs,
    )

    results = simple_evaluate(
        model=model,
        datasets=args.datasets,
        retriever=args.retriever,
        topk=args.topk,
        max_samples=args.max_samples,
        use_sliding_window=args.sliding_window,
        order=args.order,
        num_passes=args.num_passes,
        output_dir=os.path.join("outputs", time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S"))
    )

    os.makedirs(os.path.dirname(args.save_to), exist_ok=True)
    if args.save_to is not None:
        with open(args.save_to, "a") as f:
            f.write(json.dumps(
                {"args": vars(args), "results": results},
                default=str,
            ) + "\n")
        print(f"Results saved to {args.save_to}")

    print(f"encode doc time: {sum(encode_doc_times) / len(encode_doc_times):.3f}")
    print(f"encode query time: {sum(encode_query_times) / len(encode_query_times):.3f}")
