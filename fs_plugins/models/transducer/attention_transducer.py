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
from fs_plugins.modules.attention_transducer_decoder import AttentionTransducerDecoder
from fs_plugins.utils import load_pretrained_component_from_model_modified

import pdb

DEFAULT_MAX_TEXT_POSITIONS = 1024
DEFAULT_MAX_AUDIO_POSITIONS = 6000

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)



@register_model("attention_transformer_transducer")
class TransducerModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.criterion = TransducerLoss(blank=self.decoder.blank_idx)
        self.padding_idx = decoder.dictionary.pad()

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
        decoder = AttentionTransducerDecoder(args, tgt_dict, embed_tokens)
        
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
        logits, fake_src_lengths = self.decoder(prev_output_tokens, encoder_out)
        
        
        tgt_lengths = tgt_tokens.ne(self.padding_idx).sum(dim=-1)
        
        rnn_t_loss = self.criterion(logits, tgt_tokens, fake_src_lengths, tgt_lengths)
        
        
        ret_val = {
            "rnn_t_loss": {"loss": rnn_t_loss},
        }

        return ret_val
    

        

@register_model_architecture(
    "attention_transformer_transducer", "attention_transformer_transducer"
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
    "attention_transformer_transducer", "attention_t_t"
)
def t_t_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    base_architecture(args)