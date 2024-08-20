# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import Tensor
import torch.nn as nn

from examples.simultaneous_translation.utils.monotonic_attention import expected_soft_attention
#from ..utils.monotonic_attention import expected_soft_attention
from fairseq.modules import MultiheadAttention

from typing import Dict, Optional


class MonotonicAttention(MultiheadAttention):
    """
    Abstract class of monotonic attentions
    """

    def __init__(self, cfg):
        super().__init__(
            embed_dim=cfg.decoder.embed_dim,
            num_heads=cfg.decoder.attention_heads,
            kdim=cfg.encoder_embed_dim,
            vdim=cfg.encoder_embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.eps = 1e-6


    def energy_from_qk(
        self,
        query: Tensor,
        key: Tensor,
    ):
        """
        Compute energy from query and key
        q_tensor size: bsz, tgt_len, emb_dim
        k_tensor size: bsz, src_len, emb_dim
        """

        length, bsz, _ = query.size()
        q = self.q_proj.forward(query)
        q = (
            q.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        q = q * self.scaling
        length, bsz, _ = key.size()
        k = self.k_proj.forward(key)
        k = (
            k.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        energy = torch.bmm(q, k.transpose(1, 2))

        return energy



    def monotonic_attention_process_infer(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    ):
        """
        Monotonic attention at inference time
        Notice that this function is designed for simuleval not sequence_generator
        """
        assert query is not None
        assert key is not None

        soft_energy = self.energy_from_qk(
            query,
            key,
        )
        # TODO: beta_mask
        # soft_energy = soft_energy.masked_fill(beta_mask, -float("inf"))
        beta = torch.nn.functional.softmax(soft_energy, dim=-1)
    

        return beta

    def monotonic_attention_process_train(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        posterior: Optional[Tensor] = None, # posterior: B × (U+1) × T
    ):
        """
        Calculating monotonic attention process for training
        Including:
            expected soft attention: beta
        """
        assert query is not None
        assert key is not None


        soft_energy = self.energy_from_qk(
            query,
            key,
        )

        beta = expected_soft_attention(
            posterior,
            soft_energy,
            padding_mask=key_padding_mask,
            chunk_size=None,
            eps=self.eps,
        )
        
        return beta

    def forward(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        need_head_weights: bool = False,
        posterior: Optional[Tensor] = None,
    ):
        """
        query: tgt_len, bsz, embed_dim
        key: src_len, bsz, embed_dim
        value: src_len, bsz, embed_dim
        """

        assert attn_mask is None
        assert query is not None
        assert key is not None
        assert value is not None
        
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = value.size(0)

        if key_padding_mask is not None:
            assert not key_padding_mask[:, 0].any(), (
                "Only right padding is supported."
            )
            key_padding_mask = (
                key_padding_mask
                .unsqueeze(1)
                .expand([bsz, self.num_heads, src_len])
                .contiguous()
                .view(-1, src_len)
            )

        if incremental_state is not None:
            # Inference
            beta = self.monotonic_attention_process_infer(query, key, incremental_state)

        else:
            # Train
            beta = self.monotonic_attention_process_train(query, key, key_padding_mask, posterior.unsqueeze(1).expand([bsz, self.num_heads, -1, -1]).contiguous().view(-1, posterior.size(1), posterior.size(2)))

        v = self.v_proj(value)
        length, bsz, _ = v.size()
        v = (
            v.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn = torch.bmm(beta.type_as(v), v)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)
        attn_weights = None
        if need_weights:
            attn_weights = beta.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


