from typing import Optional
from dataclasses import dataclass, field

from omegaconf import II

from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass

from fs_plugins.modules.audio_convs import get_available_convs

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

@dataclass
class SpeechTransformerModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field( # type: ignore
        default="relu", 
        metadata={"help": "activation function to use"},
    )
    dropout: float = field(default=0.0, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN.",
            "alias": "--relu-dropout",
        },
    )
    relu_dropout: float = 0.0
    adaptive_input: bool = False
    
    # Relative Position
    encoder_max_relative_position: int = field(
        default=32, metadata={"help": "max_relative_position for encoder Relative attention, <0 for traditional attention"}
    )
    decoder_max_relative_position: int = field(
        default=-1, metadata={"help": "max_relative_position for decoder Relative attention, <0 for traditional attention"}
    )
    
    # Support Length
    max_audio_positions: Optional[int] = II("task.max_audio_positions")
    max_text_positions: Optional[int] = II("task.max_text_positions")
    max_source_positions: Optional[int] = II("task.max_audio_positions")
    max_target_positions: Optional[int] = II("task.max_text_positions")
   
    # Encoder Configuration
    conv_type: ChoiceEnum(get_available_convs()) = field( # type: ignore
        default= "shallow2d_base", metadata= {"help": "convolution type for speech encoder"}
    )
    encoder_embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained encoder embedding"}
    )
    encoder_embed_dim: int = field(
        default= 512, metadata={"help":"encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_layers: int = field(default=6, metadata={"help": "num encoder layers"})
    encoder_attention_heads: int = field(
        default=8, metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each encoder block"}
    )
    encoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the encoder"},
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for encoder"}
    )
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    
    # Decoder Configuration
    decoder_embed_path: Optional[str] = field(
        default=None, metadata={"help": "path to pre-trained decoder embedding"}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"}) #TODO
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"}, #TODO
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    share_all_embeddings: bool = field(
        default=False, metadata={"help":"share encoder, decoder and output embeddings (requires shared dictionary and embed dim)"}
    )
    
    
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if True, disables positional embeddings (outside self attention)"
        },
    )
    no_audio_positional_embeddings:bool = field(
        default = False,
        metadata={"help":"if True, disables positional embeddings in audio encoder"}
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute"
        },
    )
    offload_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations."
        },
    )
    
    # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
    no_cross_attention: bool = field(
        default=False, metadata={"help": "do not perform cross-attention"}
    )
    cross_self_attention: bool = field(
        default=False, metadata={"help": "perform cross+self-attention"}
    )
    
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )
    
    # Speech Encoder Configuration
    rand_pos_encoder: int = field(
        default=300,
        metadata={
            "help":"max random start for encoder position embedding"
        }
    )
    rand_pos_decoder: int = field(
        default=0,
        metadata={
            "help":"max random start for encoder position embedding"
        }
    )
    load_pretrained_encoder_from: Optional[str] = field(
        default=None, metadata={"help":"pretrained_encoder_path"}
    )
    load_pretrained_decoder_from: Optional[str] = field(
        default=None, metadata={"help":"pretrained_decoder_path"}
    )

    # params for online
    main_context:int = field(
        default=16, metadata={"help":"main context frame"}
    )
    right_context :int = field(
        default=16, metadata={"help":"right context frame"}
    )



@dataclass
class TransducerConfig(SpeechTransformerModelConfig):   
    alpha: float = field(
        default=1.0,
        metadata = {"help": "hyperparamter for constructing prior alignment"}
    )
    transducer_downsample: int = field(
        default=4,
        metadata = {"help": "source downsample ratio for transducer"}
    )
    transducer_activation: ChoiceEnum(utils.get_available_activation_fns()) = field( # type: ignore
        default="tanh", metadata={"help": "activation function to use"}
    )
    transducer_smoothing: float = field(
        default= 0., metadata = {"help":"label smoothing for transducer loss"}
    )
    tokens_per_step:int = field(
        default=20000,
        metadata={"help":"tokens per step for output head splitting"}
    )
    delay_scale:float = field(
        default=1.0,
        metadata={"help":"scale for delay loss"}
    )
    delay_func: ChoiceEnum(['zero', 'diag_positive', 'diagonal']) =field( # type: ignore
        default="diag_positive", metadata= {"help":"function for delay loss"}
    )
    transducer_ce_scale: float= field(
        default=1.0, metadata= {'help':'scale for ce loss'}
    )
    transducer_label_smoothing:float= field(
        default=0.1, metadata ={'help':"label smoothing for ce loss"}
    )
    transducer_temperature:float= field(
        default=1.0, metadata={"help":"temperature for output probs"}
    )


