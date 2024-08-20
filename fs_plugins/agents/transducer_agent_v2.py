import math
import os
import json
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
import yaml

from fairseq import checkpoint_utils, tasks, utils
from fairseq.file_io import PathManager
from examples.speech_to_text.data_utils import extract_fbank_features


from simuleval.utils import entrypoint
from simuleval.data.segments import EmptySegment, TextSegment, SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import WriteAction, ReadAction


import pdb

SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"
DEFAULT_BOS = 0
DEFAULT_EOS = 2


class OfflineFeatureExtractor:
    """
    Extract speech feature from sequence prefix.
    """

    def __init__(self, args):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.global_cmvn = args.global_cmvn
        self.device = 'cuda' if args.device == 'gpu' else 'cpu'


    def __call__(self, new_samples):
        samples = new_samples
        
        assert len(samples) >= self.num_samples_per_window
        
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )

        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )

        input_samples = samples[:effective_num_samples]

        torch.manual_seed(1)
        output = extract_fbank_features(torch.FloatTensor(input_samples).unsqueeze(0), self.sample_rate)

        output = self.transform(output)

        return torch.from_numpy(output).to(self.device)

    def transform(self, input):
        if self.global_cmvn is None:
            return input

        mean = self.global_cmvn["mean"]
        std = self.global_cmvn["std"]

        x = np.subtract(input, mean)
        x = np.divide(x, std)
        return x


class TransducerSpeechToTextAgentStates(AgentStates):

    def __init__(self, device) -> None:
        self.device = device
        self.reset()
    
    def reset(self) -> None:
        """Reset Agent states"""
        
        super().reset()
        
        self.num_complete_chunk = 0 
        self.prev_output_tokens = torch.tensor([[DEFAULT_BOS]]).long().to(self.device)

        self.unfinished_subword = []
        self.incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
        self.h_lm_last = None



@entrypoint
class TransducerSpeechToTextAgent(SpeechToTextAgent):

    speech_segment_size = 40  # in ms, 4 pooling ratio * 10 ms step size

    def __init__(self, args):
        super().__init__(args)
        
        self.device ='cuda' if args.device == 'gpu' else 'cpu'
        self.states = self.build_states()
        args.global_cmvn = None
        if args.config_yaml:
            with open(os.path.join(args.data_bin, args.config_yaml), "r") as f:
                config = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config:
                args.global_cmvn = np.load(config["global_cmvn"]["stats_npz_path"])


        self.load_model_vocab(args)
        #utils.import_user_module(args)
        
        self.feature_extractor = OfflineFeatureExtractor(args)

        self.downsample = args.transducer_downsample
        self.main_context = args.main_context
        self.right_context = args.right_context
        
        torch.set_grad_enabled(False)
        self.reset()


    def build_states(self) -> TransducerSpeechToTextAgentStates:
        """
        Build states instance for agent

        Returns:
            TransducerSpeechToTextAgentStates: agent states
        """
        return TransducerSpeechToTextAgentStates(self.device)
    
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--config-yaml", type=str, default=None,
                            help="Path to config yaml file")
        parser.add_argument("--global-stats", type=str, default=None,
                            help="Path to json file containing cmvn stats")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--user-dir", type=str, default="examples/simultaneous_translation",
                            help="User directory for simultaneous translation")
        parser.add_argument("--shift-size", type=int, default=SHIFT_SIZE,
                            help="Shift size of feature extraction window.")
        parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                            help="Window size of feature extraction window.")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--main-context", type=int, default=32)
        parser.add_argument("--right-context", type=int, default=16)
        parser.add_argument("--transducer-downsample", type=int, default=1)
        parser.add_argument("--device", type=str, default='gpu')

        # fmt: on
        return parser

    def load_model_vocab(self, args):

        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)
        utils.import_user_module(state["cfg"].common)
        
        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        if args.config_yaml is not None:
            task_args.config_yaml = args.config_yaml

        task = tasks.setup_task(task_args)

        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.model.share_memory()

        if self.device == 'cuda':
            self.model.cuda()

        # Set dictionary
        self.tgt_dict = task.target_dictionary

    @torch.inference_mode()
    def policy(self):
        
        num_frames = math.floor(
            (len(self.states.source) - self.feature_extractor.len_ms_to_samples(self.feature_extractor.window_size - self.feature_extractor.shift_size))
            / self.feature_extractor.num_samples_per_shift
        )
    
        # at least a new complete chunk is received if not finished
        if not self.states.source_finished:
            if num_frames < self.main_context * (self.states.num_complete_chunk + 1) + self.right_context:
                return ReadAction() 
        
        # this is used to caluculate self.states.num_complete_chunk
        num_complete_new_chunk = math.floor((num_frames - self.right_context) / self.main_context) - self.states.num_complete_chunk
        
        # Calculated the number of frames to make decisions
        if not self.states.source_finished:
            num_decision = num_complete_new_chunk * int(self.main_context / 4 / self.downsample)
        else:
            num_decision = None
        
        feature = self.feature_extractor(self.states.source) # prefix feature: T × C
        assert num_frames == feature.size(0)
        src_tokens = feature.unsqueeze(0) # 1 × T × C
        src_lengths = torch.tensor([feature.size(0)], device=self.device).long() # 1
    
        encoder_out = self.model.encoder(src_tokens, src_lengths)
        
        downsampled_encoder_out = encoder_out["encoder_out"][0][self.states.num_complete_chunk * (self.main_context // 4):][::self.downsample].squeeze(1) # num_decision × C
        downsampled_encoder_out = downsampled_encoder_out[:num_decision]
        
        final_output_tokens = []
            
        if self.states.h_lm_last is None:
            self.states.h_lm_last = self.model.decoder.lm(self.states.prev_output_tokens, self.states.incremental_state)[:, -1].squeeze() # V
        
        
        for i in range(downsampled_encoder_out.size(0)): 
            ii=0 # max emit per frame
            while True:
                #pdb.set_trace()
                joint_result = self.model.decoder.jointer.infer(downsampled_encoder_out[i], self.states.h_lm_last) # V
                
                log_probs = F.log_softmax(self.model.decoder.out_proj(joint_result), dim=-1)
                select_word = log_probs.argmax(dim=-1).item()
                
                if select_word == self.tgt_dict.blank_index:
                    break
                self.states.prev_output_tokens = torch.cat([self.states.prev_output_tokens, torch.tensor([[select_word]], device=self.states.prev_output_tokens.device)], dim=-1)
                final_output_tokens.append(select_word)
                self.states.h_lm_last = self.model.decoder.lm(self.states.prev_output_tokens, self.states.incremental_state)[:, -1].squeeze() # V
                ii += 1
                if ii == 5:
                    break
                #if select_word == 416:
                #    break
        
        self.states.num_complete_chunk = self.states.num_complete_chunk + num_complete_new_chunk    
        
        
        detok_output_tokens = []    
        
        for index in final_output_tokens:
            token = self.tgt_dict.string([index]) #return a string
            if token.startswith(BOW_PREFIX):
                if len(self.states.unfinished_subword) != 0:
                    detok_output_tokens += ["".join(self.states.unfinished_subword)]
                    self.states.unfinished_subword = []
                self.states.unfinished_subword += [token.replace(BOW_PREFIX, "")]
            else:    
                self.states.unfinished_subword += [token]
        
        if self.states.source_finished:
            detok_output_tokens += ["".join(self.states.unfinished_subword)]
            self.states.unfinished_subword = []    

        
        detok_output_string = " ".join(detok_output_tokens)
       
        return WriteAction(TextSegment(content=detok_output_string, finished=self.states.source_finished), finished=self.states.source_finished)   
        
            
            