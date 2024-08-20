from typing import Dict, List, Optional
import torch

from torch import Tensor
import torch.nn as nn

from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerDecoder





class AttentionDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=False
        )
        self.output_projection= None
    
    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
            for transducer, prev_output_tokens should be [bos] concat target
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
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



class AttentionTransducerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.lm = AttentionDecoder(args, dictionary, embed_tokens)
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
    
    def forward(
        self,
        prev_output_tokens:Tensor,
        encoder_out:Dict[str, List[Tensor]],
    ):
        h_lm = self.lm(prev_output_tokens, encoder_out)
       
        joint_result, fake_src_lengths = self.jointer(encoder_out, h_lm, self.padding_idx)
        
        joint_result = self.out_proj(joint_result) # it is logits, no logsoftmax performed
        
        return joint_result, fake_src_lengths
    