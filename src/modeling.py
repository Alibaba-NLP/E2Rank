from dataclasses import dataclass
import logging
from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from transformers.file_utils import ModelOutput
from transformers import PreTrainedModel, PreTrainedTokenizer, LlamaForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class TrainOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    d_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    loss_emb: Optional[Tensor] = None
    loss_ranknet: Optional[Tensor] = None


class DistributedContrastiveLoss:
    def __init__(self, temperature: float, negatives_cross_device: bool):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device        
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Cannot do negatives_cross_device without distributed training')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def __call__(self, q_reps, p_reps):
        if self.negatives_cross_device:
            # This gathers both negatives and positives.
            # It could likely be optimized by only gathering negatives.
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target *= (p_reps.size(0) // q_reps.size(0))
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None: return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        # All tensors have the same shape, as pooling already applied to them
        dist.all_gather(all_tensors, t)

        all_tensors[self.rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return self.cosine_similarity(q_reps, p_reps)

    def cosine_similarity(self, q_reps, d_reps):
        if len(d_reps.size()) == 2:
            return torch.matmul(q_reps, d_reps.transpose(0, 1))
        return torch.matmul(q_reps, d_reps.transpose(-2, -1))


def rank_net(y_pred, y_true, weighted=False, use_rank=False, weight_by_diff=False, weight_by_diff_powed=False):
    if use_rank is None:
        y_true = torch.tensor([[1 / (np.argsort(y_true)[::-1][i] + 1) for i in range(y_pred.size(1))]] * y_pred.size(0)).cuda()

    # generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weighted:
        values, indices = torch.sort(y_true, descending=True)
        ranks = torch.zeros_like(indices)
        ranks.scatter_(1, indices, torch.arange(1, y_true.numel() + 1).to(y_true.device).view_as(indices))
        pairs_ranks = ranks[:, document_pairs_candidates] 
        rank_sum = pairs_ranks.sum(-1)
        weight = 1 / rank_sum[the_mask] # Relevance Feedback
        # rank_prod=pairs_ranks[:, :, 0]*pairs_ranks[:, :, 1]
        # weight = rank_sum[the_mask]/rank_prod[the_mask]      
    else:
        if weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

    # 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor,
    normalize_embeddings: bool = True
) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        embeddings = last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        embeddings = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    if normalize_embeddings:
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)
    return embeddings


class JointTrainingModel(torch.nn.Module):

    def __init__(
        self,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer = None,
        use_embed_loss: bool = True,
        temperature: float = 0.03,
        use_ranknet_loss: bool = False,
        negatives_cross_device: bool = False,
        loss_embed_factor: float = 1.0,
        loss_ranknet_factor: float = 1.0,
        ranknet_scale_factor: float = 5.0,
    ):
        super().__init__()
        self.model = model
        self.config = self.model.config # Required for accelerate DeepSpeed integration

        self.use_embed_loss = use_embed_loss
        self.emb_loss_fn = DistributedContrastiveLoss(temperature, negatives_cross_device)
        self.loss_embed_factor = loss_embed_factor

        self.use_ranknet_loss = use_ranknet_loss
        if self.use_ranknet_loss:
            self.ranking_loss_fn = rank_net
            self.loss_ranknet_factor = loss_ranknet_factor
            self.ranknet_scale_factor = ranknet_scale_factor

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        document: Dict[str, torch.Tensor] = None,
        pseudo_query: Dict[str, torch.Tensor] = None,
        ranking: torch.Tensor = None
    ):

        q_reps = last_token_pool(
            self.model(**query).last_hidden_state,
            query['attention_mask'],
            normalize_embeddings=True,
        )

        d_reps = last_token_pool(
            self.model(**document).last_hidden_state,
            document['attention_mask'],
            normalize_embeddings=True,
        )

        if self.use_embed_loss:
            d_reps_for_embedding_loss = d_reps.reshape(q_reps.size(0), -1, q_reps.size(-1))
            if ranking is not None:
                d_reps_for_embedding_loss = torch.gather(d_reps_for_embedding_loss, dim=1, index=ranking.unsqueeze(-1).expand(-1, -1, q_reps.size(-1)))
            d_reps_for_embedding_loss = d_reps_for_embedding_loss.reshape(-1, q_reps.size(-1))
            loss_emb = self.emb_loss_fn(
                q_reps,
                d_reps_for_embedding_loss
            ) * self.loss_embed_factor
        else:
            loss_emb = None

        if self.use_ranknet_loss:
            if pseudo_query is not None:
                pseudo_q_reps = last_token_pool(
                    self.model(**pseudo_query).last_hidden_state,
                    pseudo_query['attention_mask'],
                    normalize_embeddings=True,
                ).unsqueeze(1)  # (B, 1, D)
            else:
                pseudo_q_reps = q_reps.unsqueeze(1)

            batch_size, slate_length = ranking.shape
            scores = pseudo_q_reps @ d_reps.reshape(batch_size, slate_length, -1).transpose(-1, -2)
            scores = scores.squeeze(1) * self.ranknet_scale_factor  # scale factor

            rank_position = torch.empty_like(ranking, device=ranking.device, dtype=torch.long)
            rank_indices = torch.arange(slate_length, device=ranking.device).expand(batch_size, -1)
            rank_position.scatter_(dim=1, index=ranking, src=rank_indices)

            loss_ranknet = self.ranking_loss_fn(scores, slate_length - rank_position)  # use -rank_position as relevance
            loss_ranknet = loss_ranknet * self.loss_ranknet_factor
        else:
            loss_ranknet = None

        loss = sum([x for x in [loss_emb, loss_ranknet] if x is not None])

        return TrainOutput(
            loss=loss,
            loss_emb=loss_emb,
            loss_ranknet=loss_ranknet
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

