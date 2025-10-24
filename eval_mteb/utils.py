import json
import os
import mteb
import os
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
import csv

from mteb.tasks.Retrieval.multilingual.MLQARetrieval import _LANGUAGES, _build_lang_pair

def load_rarb_data(self, **kwargs):
    RARB_prefix="./mtebdata/RAR-b/"
    if self.data_loaded:
        return
    self.corpus, self.queries, self.relevant_docs = {}, {}, {}
    dataset_path = self.metadata_dict["dataset"]["path"]
    path = os.path.join(RARB_prefix, dataset_path.split('/')[-1])
    qrel_path = os.path.join(path, "qrels/test.tsv")
    split = "test"
    self.relevant_docs[split] = {} 
    with open(qrel_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader) # skip header row
        for i, row in enumerate(reader):
            qid, did, score = row[0], row[1], int(row[2])
            if qid not in self.relevant_docs[split]:
                self.relevant_docs[split][qid] = {}
            self.relevant_docs[split][qid][did] = score
    filepath = os.path.join(path, "corpus.jsonl")
    self.corpus[split] = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            self.corpus[split][line['_id']] = line['title'] + line['text']
    
    filepath = os.path.join(path, "corpus.jsonl")
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            self.corpus[line['_id']] = line['title'] + line['text']
    filepath = os.path.join(path, "queries.jsonl")
    self.queries[split] = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            self.queries[split][line['_id']] = line['text']

    self.data_loaded = True
    
def load_mlqa_data(self, **kwargs):
    """In this retrieval datasets, corpus is in lang XX and queries in lang YY."""
    if self.data_loaded:
        return
    _dataset_raw = {}
    self.queries, self.corpus, self.relevant_docs = {}, {}, {}
    save_path = './mtebdata/rank_test_data/MLQA/'
    for hf_subset, langs in _LANGUAGES.items():
        # Builds a language pair separated by an underscore. e.g., "ara-Arab_eng-Latn".
        # Corpus is in ara-Arab and queries in eng-Latn
        lang_pair = _build_lang_pair(langs)

        self.queries[lang_pair] = {}
        self.corpus[lang_pair] = {}
        self.relevant_docs[lang_pair] = {}

        for eval_split in self.metadata.eval_splits:
            with open(os.path.join(save_path, f'{lang_pair}_{eval_split}_queries.json')) as f:
                self.queries[lang_pair][eval_split] = json.load(f)
            with open(os.path.join(save_path, f'{lang_pair}_{eval_split}_corpus.json')) as f:
                self.corpus[lang_pair][eval_split] = json.load(f)
            with open(os.path.join(save_path, f'{lang_pair}_{eval_split}_relevant_docs.json')) as f:
                self.relevant_docs[lang_pair][eval_split] = json.load(f)
    self.data_loaded = True

def load_belebel_data(self, **kwargs):
    """In this retrieval datasets, corpus is in lang XX and queries in lang YY."""
    if self.data_loaded:
        return
    _dataset_raw = {}
    self.queries, self.corpus, self.relevant_docs = {}, {}, {}
    save_path = './mtebdata/BelebeleRetrieval/'
    lang_pairs = []
    with open(os.path.join(save_path, 'lang_pairs.txt')) as f:
        for line in f:
            line = line.strip()
            lang_pairs.append(line) 
    # for hf_subset, langs in _LANGUAGES.items():
    for lang_pair in lang_pairs:
        # Builds a language pair separated by an underscore. e.g., "ara-Arab_eng-Latn".
        # Corpus is in ara-Arab and queries in eng-Latn
        self.queries[lang_pair] = {}
        self.corpus[lang_pair] = {}
        self.relevant_docs[lang_pair] = {}

        for eval_split in self.metadata.eval_splits:
            with open(os.path.join(save_path, f'{lang_pair}_{eval_split}_queries.json')) as f:
                self.queries[lang_pair][eval_split] = json.load(f)
            with open(os.path.join(save_path, f'{lang_pair}_{eval_split}_corpus.json')) as f:
                self.corpus[lang_pair][eval_split] = json.load(f)
            with open(os.path.join(save_path, f'{lang_pair}_{eval_split}_relevant_docs.json')) as f:
                self.relevant_docs[lang_pair][eval_split] = json.load(f)
    self.data_loaded = True

def load_hagrid_data(self,  **kwargs):
    if self.data_loaded:
        return
    data_path = './mtebdata/hagrid/'
    with open(os.path.join(data_path, f"{self.metadata.eval_splits[0]}_queries.json")) as f:
        self.queries = {
            self.metadata.eval_splits[0]: json.load(f)
        }
    with open(os.path.join(data_path, f"{self.metadata.eval_splits[0]}_corpus.json")) as f:
        self.corpus = {
           self.metadata.eval_splits[0]: json.load(f)
        }
    with open(os.path.join(data_path, f"{self.metadata.eval_splits[0]}_relevant_docs.json")) as f:
        self.relevant_docs = {
            self.metadata.eval_splits[0]: json.load(f)
        }
    self.data_loaded = True
