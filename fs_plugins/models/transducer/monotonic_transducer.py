# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from pathlib import Path


import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq import utils, checkpoint_utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import Embedding




import logging
logger = logging.getLogger(__name__)

from fs_plugins.models.transducer.transducer_config import TransducerConfig
from fs_plugins.models.transducer.transducer_loss import TransducerLoss
from fs_plugins.modules.unidirectional_encoder import UnidirectionalAudioTransformerEncoder
from fs_plugins.modules.monotonic_transducer_decoder import MonotonicTransducerDecoder
from fs_plugins.utils import load_pretrained_component_from_model_modified

import pdb

DEFAULT_MAX_TEXT_POSITIONS = 1024
DEFAULT_MAX_AUDIO_POSITIONS = 6000

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)



@register_model("monotonic_transformer_transducer")
class TransducerModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.criterion = TransducerLoss(blank=self.decoder.blank_idx)
        self.padding_idx = decoder.dictionary.pad()
        self.attn_step = int(self.encoder.main_context / self.decoder.downsample)

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransducerConfig(), delete_default=True, with_prefix=""
        )
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        # base_architecture(args)
        
        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_AUDIO_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TEXT_POSITIONS
        

        decoder_embed_tokens = cls.build_embedding(
            args, task.target_dictionary, args.decoder_embed_dim
        )

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task.target_dictionary, decoder_embed_tokens)
        
        model = cls(args, encoder, decoder)
        
        return model
    
    
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb
    

    @classmethod
    def build_encoder(cls, args):
        encoder = UnidirectionalAudioTransformerEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped loading pretrained encoder because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")

        return encoder
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = MonotonicTransducerDecoder(args, tgt_dict, embed_tokens)
        
        pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped loading pretrained decoder because {pretraining_path} does not exist"
                )
            else:
                #decoder = checkpoint_utils.load_pretrained_component_from_model(
                #    component=decoder, checkpoint=pretraining_path, strict=False
                #)
                decoder, keys_info = load_pretrained_component_from_model_modified(
                    component=decoder, checkpoint=pretraining_path, strict=False
                )
                logger.info(f"loaded pretrained decoder from: {pretraining_path}")
                logger.info(f"keys information: {keys_info}")
        return decoder


    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens
    ):       
        encoder_out = self.encoder(src_tokens, fbk_lengths=src_lengths)
        
        src_lengths = (~encoder_out["encoder_padding_mask"][0]).sum(dim=-1)
        tgt_lengths = tgt_tokens.ne(self.padding_idx).sum(dim=-1)
        
        with torch.no_grad():
            frame_prior = torch.ones([src_lengths.size(0), tgt_lengths.max()+1, src_lengths.max()]).to(encoder_out["encoder_out"][0]) # B, U+1, T
            frame_prior = frame_prior.masked_fill(encoder_out["encoder_padding_mask"][0].unsqueeze(1), 0)
            frame_prior = frame_prior / src_lengths.unsqueeze(-1).unsqueeze(-1)
        
            logits, fake_src_lengths = self.decoder(prev_output_tokens, encoder_out, frame_prior)
            rnn_t_loss, posterior = self.criterion.forward_w_posterior(logits, tgt_tokens, fake_src_lengths, tgt_lengths)
            assert not torch.isnan(posterior).any()
            assert not torch.isinf(posterior).any()
            
        # 假设我们需要填充到的最大长度
        max_tgt_length = tgt_lengths.max().item()

        # 构造 padding_mask
        batch_size = tgt_lengths.size(0)
        tgt_padding_mask = torch.arange(max_tgt_length).expand(batch_size, max_tgt_length).to(tgt_lengths) >= tgt_lengths.unsqueeze(1)

        
        with torch.no_grad():
            T = logits.size(1)
            bos_posterior = torch.zeros(T, dtype=posterior.dtype, device=posterior.device)
            # only support right padding
            bos_posterior[0] = 1.0
            
            exp_posterior = torch.exp(posterior) # B, T, U
            #exp_posterior.masked_fill_(tgt_padding_mask.unsqueeze(1), 1/(src_lengths.max()))
            posterior = torch.cat([bos_posterior.unsqueeze(0).unsqueeze(-1).expand(posterior.size(0), -1, -1), exp_posterior], dim=-1) # B, T, U+1
            
            if encoder_out["encoder_padding_mask"][0].any():
                posterior.masked_fill_(encoder_out["encoder_padding_mask"][0][:,::self.decoder.downsample].unsqueeze(-1),0)
                
            if posterior.size(1) <= self.attn_step:
                chunk_posterior = posterior.sum(dim=1, keepdim=True)
            else:    
                pad_length = self.attn_step - posterior.size(1) % self.attn_step
                if pad_length == self.attn_step:
                    pad_length = 0
                padded_posterior = torch.nn.functional.pad(posterior, (0, 0, 0, pad_length), mode='constant', value=0)
                B, T, U = padded_posterior.size()
                chunk_posterior = padded_posterior.view(B, -1, self.attn_step, U).sum(dim=2)
            
            #check: posterior[b,:,:].argmax(dim=0)[:tgt_lengths[b]]
            #check: chunk_posterior[-1].sum(dim=0)
            
            chunk_posterior = chunk_posterior.transpose(1,2) # change it to B, U+1, T to be compatible with mono_attn
            
            frame_posterior = None
            if chunk_posterior.size(2) > 1: # T > 1    
                frame_posterior_pad = torch.zeros([chunk_posterior.size(0), chunk_posterior.size(1), chunk_posterior.size(2)-1, self.encoder.main_context-1], dtype=posterior.dtype, device=posterior.device)
                frame_posterior = torch.cat([frame_posterior_pad, chunk_posterior[:,:,:-1].unsqueeze(-1)], dim=-1) # B, U+1, T-1, self.encoder.main_context
                frame_posterior = frame_posterior.view(frame_posterior.size(0), frame_posterior.size(1), -1) # B, U+1, T-1*self.encoder.main_context
        
            tail_lengths = src_lengths - (chunk_posterior.size(2) - 1) * self.encoder.main_context
            
            frame_posterior_tail = torch.zeros([chunk_posterior.size(0), chunk_posterior.size(1), tail_lengths.max()], dtype=posterior.dtype, device=posterior.device)
            
            # torch.clamp: handle those tail_lengths < 0
            frame_posterior_tail.scatter_(dim=-1, index=torch.clamp((tail_lengths-1),min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, chunk_posterior.size(1),-1), src=chunk_posterior[:,:,-1:])
            
            if frame_posterior is not None:
                frame_posterior = torch.cat([frame_posterior, frame_posterior_tail], dim=-1) # B, U+1, enc_length
            else:
                frame_posterior = frame_posterior_tail
            
            frame_posterior[:,1:,:].masked_fill_(tgt_padding_mask.unsqueeze(-1), 1/(src_lengths.max()))    
            assert not torch.isnan(frame_posterior).any()
            assert not torch.isinf(frame_posterior).any()
            
        monotonic_logits, monotonic_fake_src_lengths = self.decoder(prev_output_tokens, encoder_out, frame_posterior)
        
        montonic_rnn_t_loss = self.criterion(monotonic_logits, tgt_tokens, monotonic_fake_src_lengths, tgt_lengths)
        assert not torch.isnan(montonic_rnn_t_loss).any()
        assert not torch.isinf(montonic_rnn_t_loss).any()
        ret_val = {
            #"rnn_t_loss": {"loss": rnn_t_loss},
            "montonic_rnn_t_loss": {"loss": montonic_rnn_t_loss, "factor": 1.0},
        }

        return ret_val
    

        

@register_model_architecture(
    "monotonic_transformer_transducer", "monotonic_transformer_transducer"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- speech arguments ---
    args.rand_pos_encoder = getattr(args, "rand_pos_encoder", 300)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.conv_type= getattr(args, "conv_type", "shallow2d_base")
    args.no_audio_positional_embeddings = getattr(
        args, "no_audio_positional_embeddings", False
    )
    args.main_context = getattr(args, "main_context", 32)
    args.right_context = getattr(args, "right_context", 16)
    args.encoder_max_relative_position = getattr(args, "encoder_max_relative_position", 32)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.transducer_downsample = getattr(args, "transducer_downsample", 1)
    
@register_model_architecture(
    "monotonic_transformer_transducer", "monotonic_t_t"
)
def t_t_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    base_architecture(args)