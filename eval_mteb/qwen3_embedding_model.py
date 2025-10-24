from __future__ import annotations

import logging
import queue
import json
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

from tqdm.autonotebook import tqdm
import numpy as np
import torch
from torch.utils.data._utils.worker import ManagerWatchdog
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import BatchEncoding
from peft import AutoPeftModelForCausalLM, PeftModel
from mteb.encoder_interface import PromptType
from mteb.models.wrapper import Wrapper
from mteb.model_meta import ModelMeta
import mteb

logger = logging.getLogger(__name__)


class TransformersTextEmbedder(torch.nn.Module):
    def __init__(
        self,
        model: str,
        pooler_type: str = 'last',
        do_norm: bool = False,
        truncate_dim: int = 0,
        padding_left: bool = False,
        attn_type: str = 'causal',
        **kwargs,
    ):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model, **kwargs).model  # hotfix
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.tokenizer.padding_side = "left"
        self.pooler_type = pooler_type
        self.do_norm = do_norm
        self.truncate_dim = truncate_dim
        self.padding_left = padding_left
        self.attn_type = attn_type
        if pooler_type == 'first':
            assert padding_left is False
            self.pooling = self._pooling_first
        elif pooler_type == 'last':
            self.pooling = self._pooling_last
            
        elif pooler_type == 'mean':
            self.pooling = self._pooling_mean
        
        else:
            ValueError(f"Wrong pooler : {self.pooler_type}")

    def embed(
        self, 
        sentences: Sequence[str], 
        max_length: int,
        prompt: str | None = None,
        device: str | torch.device = 'cpu',
    ) -> torch.Tensor:
        inputs = self.tokenize(sentences, max_length, prompt).to(device)
        embeddings = self.forward(**inputs.data)
        return embeddings

    def tokenize(self, texts, max_length: int, prompt=None) -> BatchEncoding:
        if prompt:
            texts = [prompt + t for t in texts]
        texts = [t + "<|endoftext|>" for t in texts]  # add eos token, which is different from original code, since we modify the tokenizer
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        return inputs

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        embeddings = self.pooling(output.last_hidden_state, attention_mask)
        if self.truncate_dim > 0:
            embeddings = embeddings[:, :self.truncate_dim]
        if self.do_norm:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask
        left_padding = (mask[:, -1].sum() == mask.shape[0])
        if left_padding:
            return hidden_state[:, -1]
        else:
            sequence_lengths = mask.sum(dim=1) - 1
            batch_size = hidden_state.shape[0]
            return hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths]

    @staticmethod
    def _pooling_first(hidden_state: torch.Tensor, _) -> torch.Tensor:
        return hidden_state[:, 0]

    @staticmethod
    def _pooling_last_left(hidden_state: torch.Tensor, _) -> torch.Tensor:
        return hidden_state[:, -1]

    @staticmethod
    def _pooling_last_right(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_indices = attention_mask.sum(1) - 1
        batch_indices = torch.arange(hidden_state.size(0), device=hidden_state.device)
        pooled_output = hidden_state[batch_indices, last_indices]
        return pooled_output

    @staticmethod
    def _pooling_mean(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert attention_mask.ndim == 2, f"Unexpected {attention_mask.ndim=}"
        attention_mask = attention_mask.float()
        lengths = attention_mask.sum(1)
        pooled_output = torch.einsum('bsh,bs,b->bh', (hidden_state.float(), attention_mask, 1 / lengths))
        return pooled_output


def _encode_loop(
    model: TransformersTextEmbedder,
    input_queue,
    output_queue,
    device: torch.device,
    qsize: int = 4,
    amp_dtype=None
):
    model = model.to(device)
    watchdog = ManagerWatchdog()
    keep_queue = queue.Queue(qsize + 1)

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype
        ) if amp_dtype is not None else nullcontext():
            while watchdog.is_alive():
                r = input_queue.get()
                if r is None:
                    break

                n, inputs = r
                embeddings = model.embed(*inputs, device=device)
                output_queue.put((n, embeddings))
                if keep_queue.full():
                    i = keep_queue.get()
                    del i
                keep_queue.put(embeddings)
                del r, n, inputs

    while not keep_queue.empty():
        i = keep_queue.get()
        del i
    del model, watchdog
    return


class Qwen3Embedding(Wrapper):
    _model_class = TransformersTextEmbedder
    # `_model_class` needs to implement `embed(batch, max_length, prompt_name, self.device)`.

    def __init__(
        self,
        model: str,
        use_instruction: bool = False,
        device: str = 'cuda',
        max_length: int = 512,
        max_query_length: int | None = None,
        max_doc_length: int | None = None,
        precision: str = 'fp32',
        mp_qsize: int = 4,
        instruction_dict_path=None,
        instruction_template=None, 
        **kwargs,  # For `TransformersTextEmbedder`
    ) -> None:
        
        model_name = model.split('/')
        if model_name[-1] == '':
            model_name = model_name[-2]
        else:
            model_name = model_name[-1]
        model_name = kwargs.pop('model_name', model_name)
        self.model = self._model_class(model, **kwargs)
        self.mteb_model_meta = ModelMeta(
            name=model_name, revision=kwargs.get('revision', None), release_date=None, languages=None, n_parameters=None, memory_usage_mb=None, max_tokens=None, embed_dim=None, license=None, open_weights=False, public_training_code=None, public_training_data=None, framework=["Sentence Transformers"], similarity_fn_name="cosine", use_instructions=True, training_datasets=None
        )

        self.use_instruction = use_instruction
        self.device = device
        self.max_query_length = max_query_length or max_length
        self.max_doc_length = max_doc_length or max_length
        self.amp_dtype = None
        if precision == 'fp16':
            self.model.half()
        elif precision == 'bf16':
            self.model.bfloat16()
        elif precision.startswith('amp_'):
            self.amp_dtype = torch.float16 if precision.endswith('fp16') else torch.bfloat16
        self.mp_qsize = mp_qsize
        n_gpu = torch.cuda.device_count()
        self.world_size = n_gpu
        assert n_gpu > 0, 'woho, no no no!'
        logger.info(f"We have {n_gpu=}, good.")
        self._input_queues = list()
        self._output_queues = list()
        self._workers = list()
        self.instruction_dict = dict()
        if instruction_dict_path is not None:
            instruction_dict_path = instruction_dict_path
            with open(instruction_dict_path) as f:
                self.instruction_dict = json.load(f)
        if instruction_template is not None:
            self.instruction_template = instruction_template

    def get_instruction(self, task_name, prompt_type):
        sym_task = False
        if task_name in self.instruction_dict:
            instruction = self.instruction_dict[task_name]
            if isinstance(instruction, dict):
                instruction = instruction.get(prompt_type, "")
                sym_task = True
        else:
            instruction = super().get_instruction(task_name, prompt_type)
        try:
            task_type = mteb.get_tasks(tasks=[task_name])[0].metadata.type
        except:
            task_type = "Retrieval"  # hotfix for additional retrieval tasks
        if 'Retrieval' in task_type and not sym_task and prompt_type != 'query':
            return ""
        if task_type in ["STS", "PairClassification"]:
            return "Retrieve semantically similar text"
        if task_type in "Bitext Mining":
            return "Retrieve parallel sentences"
        if 'Retrieval' in task_type and prompt_type == 'query' and instruction is None:
            instruction = "Retrieval relevant passage for the given query."
        return instruction
        
    def format_instruction(self, instruction, prompt_type):
        if instruction is not None and len(instruction.strip()) > 0:
            instruction = self.instruction_template.format(instruction)
            return instruction
        return ""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        instruction = None
        if self.use_instruction:
            instruction = self.get_instruction(task_name, prompt_type)
            if self.instruction_template:
                instruction = self.format_instruction(instruction, prompt_type)
            logger.info(f"Using instruction: '{instruction}' for task: '{task_name}'")

        num_texts = len(sentences)
        logger.info(f"Encoding {num_texts} sentences.")
        num_batches = num_texts // batch_size + int(num_texts % batch_size > 0)

        def _receive(oq, timeout=0.00125):
            try:
                n, embed = oq.get(timeout=timeout)
                result_dict[n] = embed.cpu()
                pbar.update(1)
                del embed
            except queue.Empty:
                pass

        max_length = self.max_query_length if prompt_type == PromptType.query else self.max_doc_length

        pbar = tqdm(
            total=num_batches, disable=not show_progress_bar, desc='encode',
            mininterval=1, miniters=10
        )
        result_dict = dict()
        if not self._workers:
            self.model.to(self.device)

        with nullcontext() if self._workers else torch.inference_mode():
            with nullcontext() if self._workers or self.amp_dtype is None else torch.autocast(
                device_type=self.device, dtype=self.amp_dtype
            ):
                for n, i in enumerate(range(0, num_texts, batch_size)):
                    batch = sentences[i: i + batch_size]
                    if self._workers:
                        rank = n % self.world_size
                        self._input_queues[rank].put((n, (batch, max_length, instruction)))
                        if n >= self.world_size:
                            _receive(self._output_queues[rank])
                    else:
                        result_dict[n] = self.model.embed(batch, max_length, instruction, self.device)
                        pbar.update(1)
        if self._workers:
            while len(result_dict) < num_batches:
                for oq in self._output_queues:
                    _receive(oq)

        pbar.close()
        results = [result_dict[n] for n in range(len(result_dict))]
        embeddings = torch.cat(results).float()
        assert embeddings.shape[0] == num_texts
        embeddings = embeddings.cpu().numpy()
        return embeddings


    def start(self):
        self.model.share_memory()
        logger.warning(f"Starting {self.world_size} worker processes.")
        mp_ctx = torch.multiprocessing.get_context('spawn')
        self._input_queues = [mp_ctx.Queue(self.mp_qsize) for _ in range(self.world_size)]
        self._output_queues = [mp_ctx.Queue(self.mp_qsize) for _ in range(self.world_size)]
        self._workers = list()
        for i, (iq, oq) in enumerate(zip(self._input_queues, self._output_queues)):
            device = torch.device(f'cuda:{i}')
            encode_worker = mp_ctx.Process(
                target=_encode_loop, name=f'encode_{i}', args=(
                    self.model, iq, oq, device, self.mp_qsize, self.amp_dtype
                )
            )
            encode_worker.start()
            self._workers.append(encode_worker)
            logger.warning(f"GPU {i} worker initiated.")

    def stop(self):
        [q.put(None) for q in self._input_queues]
        [w.join() for w in self._workers]
        [w.close() for w in self._workers]
        for qs in (self._input_queues, self._output_queues):
            [q.put(None) for q in qs]
