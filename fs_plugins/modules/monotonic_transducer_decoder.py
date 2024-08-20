from typing import Dict, List, Optional
import torch

from torch import Tensor
import torch.nn as nn

from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerDecoder
from fairseq.modules import LayerNorm
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap
import pdb
from .monotonic_transformer_layer import MonotonicTransformerDecoderLayer




class MonotonicDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=False
        )
        self.output_projection= None
        
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = MonotonicTransformerDecoderLayer(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
        
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        posterior: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            posterior,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        posterior: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                posterior,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}
    
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        posterior: Tensor = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
            for transducer, prev_output_tokens should be [bos] concat target (including [eos])
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            posterior=posterior,
            incremental_state=incremental_state,
            full_context_alignment=False,
            alignment_layer=None,
            alignment_heads=None,
        )
        return x


class AddJointNet(nn.Module):
    def __init__(
        self,
        encoder_dim,
        decoder_dim, 
        hid_dim, 
        activation="tanh",
        downsample=1,
    ):
        super().__init__()
        self.downsample = downsample
        self.encoder_proj = nn.Linear(encoder_dim, hid_dim)
        self.decoder_proj = nn.Linear(decoder_dim, hid_dim)
        self.activation_fn = utils.get_activation_fn(activation)
        #self.joint_proj = nn.Linear(hid_dim, hid_dim)
        #self.layer_norm = LayerNorm(hid_dim)
        if downsample < 1:
            raise ValueError("downsample should be more than 1 for add_joint")
        
    def forward(self, encoder_out:Dict[str, List[Tensor]], decoder_state, padding_idx):
        """
            use dimension same as transformer
            Args:
            encoder_out: "encoder_out": TxBxC
            decoder_state: BxUxC
        """
        encoder_state = encoder_out["encoder_out"][0]
        encoder_state = encoder_state[::self.downsample].contiguous()
        encoder_state = encoder_state.transpose(0,1)

        h_enc = self.encoder_proj(encoder_state)
        h_dec = self.decoder_proj(decoder_state)
        h_joint = h_enc.unsqueeze(2) + h_dec.unsqueeze(1)
        h_joint = self.activation_fn(h_joint)
        #h_joint = self.joint_proj(h_joint)
        #h_joint = self.layer_norm(h_joint)
        
        fake_src_tokens = (encoder_out["encoder_padding_mask"][0]).long()
        fake_src_lengths = fake_src_tokens.ne(padding_idx).sum(dim=-1)
        fake_src_lengths = (fake_src_lengths / self.downsample).ceil().long()
        
        return h_joint, fake_src_lengths
    
    def infer(self, encoder_state, decoder_state):
        """
            use dimension same as transformer
            Args:
            encoder_out: "encoder_out": C
            decoder_state: C
        """

        h_enc = self.encoder_proj(encoder_state)
        h_dec = self.decoder_proj(decoder_state)
        h_joint = h_enc + h_dec
        h_joint = self.activation_fn(h_joint)
        #h_joint = self.joint_proj(h_joint)
        #h_joint = self.layer_norm(h_joint)
        
        return h_joint
    
    

class ConcatJointNet(nn.Module):
    def __init__(
            self,
            encoder_dim,
            decoder_dim,
            hid_dim, 
            activation="tanh",
            downsample=1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear((encoder_dim+decoder_dim), hid_dim)
        self.downsample = downsample
        self.activation_fn = utils.get_activation_fn(activation)
        if downsample < 1:
            raise ValueError("downsample should be more than 1 for concat_joint")
    
    def forward(self, encoder_out:Dict[str, List[Tensor]], decoder_state, padding_idx):
    
        encoder_state = encoder_out["encoder_out"][0]
        encoder_state = encoder_state[::self.downsample].contiguous() #TODO: downsample
        encoder_state = encoder_state.transpose(0,1)
        
        seq_lens = encoder_state.size(1)
        target_lens = decoder_state.size(1)
        
        encoder_state = encoder_state.unsqueeze(2)
        decoder_state = decoder_state.unsqueeze(1)
        
        encoder_state = encoder_state.expand(-1, -1, target_lens, -1)
        decoder_state = decoder_state.expand(-1, seq_lens, -1, -1)
        
        h_joint = torch.cat((encoder_state, decoder_state), dim=-1)

        h_joint = self.fc1(h_joint)
        h_joint = self.activation_fn(h_joint)
        
        fake_src_tokens = (encoder_out["encoder_padding_mask"][0]).long()
        fake_src_lengths = fake_src_tokens.ne(padding_idx).sum(dim=-1)
        fake_src_lengths = (fake_src_lengths / self.downsample).ceil().long()

        return h_joint, fake_src_lengths



class MonotonicTransducerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.lm = MonotonicDecoder(args, dictionary, embed_tokens)
        self.output_embed_dim = args.decoder_output_dim
        self.out_proj = nn.Linear(args.decoder_output_dim, len(dictionary), bias=False)
        if args.share_decoder_input_output_embed:
            self.out_proj.weight= embed_tokens.weight
        else:
            nn.init.normal_(
                self.out_proj.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        self.blank_idx= dictionary.blank_index
        self.padding_idx = dictionary.pad()
        self.downsample = getattr(args, "transducer_downsample", 1)
        #self.jointer = ConcatJointNet(args.encoder_embed_dim, args.decoder_output_dim, args.decoder_output_dim, downsample=self.downsample)
        self.jointer = AddJointNet(args.encoder_embed_dim, args.decoder_output_dim, args.decoder_output_dim, downsample=self.downsample)
    
    def forward_naive(
        self,
        prev_output_tokens:Tensor,
        encoder_out:Dict[str, List[Tensor]],
    ):
        h_lm = self.lm(prev_output_tokens)
       
        joint_result, fake_src_lengths = self.jointer(encoder_out, h_lm, self.padding_idx)
        
        joint_result = self.out_proj(joint_result) # it is logits, no logsoftmax performed
        
        return joint_result, fake_src_lengths
    
    def forward(
        self,
        prev_output_tokens:Tensor,
        encoder_out:Dict[str, List[Tensor]],
        posterior:Tensor,
    ):
        h_lm = self.lm(prev_output_tokens, encoder_out, posterior)
       
        joint_result, fake_src_lengths = self.jointer(encoder_out, h_lm, self.padding_idx)
        
        joint_result = self.out_proj(joint_result) # it is logits, no logsoftmax performed
        
        return joint_result, fake_src_lengths
    