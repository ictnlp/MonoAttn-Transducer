# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import torch
import logging
import numpy as np

from argparse import Namespace
from fairseq import utils, metrics
from fairseq.tasks import register_task
from fairseq.utils import new_arange
from fairseq.optim.amp_optimizer import AMPOptimizer


from fairseq.tasks.speech_to_text import SpeechToTextTask

from fs_plugins.datasets.transducer_speech_to_text_dataset import (
    TransducerSpeechToTextDataset,
    TransducerSpeechToTextDatasetCreator,
)


logger = logging.getLogger(__name__)
EVAL_BLEU_ORDER = 4

@register_task("transducer_speech_to_text")
class TransducerSpeechToTextTask(SpeechToTextTask):

    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        self.pre_tokenizer = self.build_tokenizer(self.args)
        self.bpe_tokenizer = self.build_bpe(self.args)
        blank_index = self.tgt_dict.add_symbol("<blank>")
        self.tgt_dict.blank_index = blank_index

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        if self.args.eval_bleu:
            gen_args = json.loads(self.args.eval_bleu_args)
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    @classmethod
    def add_args(cls, parser):
        SpeechToTextTask.add_args(parser)
        
        
        # options for reporting BLEU during validation
        parser.add_argument(
            "--eval-bleu",
            action="store_true",
            help="evaluation with BLEU scores",
        )
        parser.add_argument(
            "--eval-bleu-detok",
            type=str,
            default="space",
            help="detokenize before computing BLEU (e.g., 'moses'); "
                 "required if using --eval-bleu; use 'space' to "
                 "disable detokenization; see fairseq.data.encoders "
                 "for other options",
        )
        parser.add_argument(
            "--eval-bleu-detok-args",
            type=str,
            metavar="JSON",
            help="args for building the tokenizer, if needed",
        )
        parser.add_argument(
            "--eval-tokenized-bleu",
            action="store_true",
            default=False,
            help="compute tokenized BLEU instead of sacrebleu",
        )
        parser.add_argument(
            "--eval-bleu-remove-bpe",
            nargs="?",
            const="@@ ",
            default=None,
            help="remove BPE before computing BLEU",
        )
        parser.add_argument(
            "--eval-bleu-args",
            type=str,
            metavar="JSON",
            help="generation args for BLUE scoring, "
                 "e.g., '{\"beam\": 4, \"lenpen\": 0.6}'",
        )
        parser.add_argument(
            "--eval-bleu-print-samples",
            action="store_true",
            help="print sample generations during validation",
        )
        parser.add_argument(
            "--eval-bleu-bpe",
            type=str,
            metavar="BPE",
            default=None,
            help="args for building the bpe, if needed",
        )
        parser.add_argument(
            "--eval-bleu-bpe-path",
            type=str,
            metavar='BPE',
            help="args for building the bpe, if needed",
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = TransducerSpeechToTextDatasetCreator.from_tsv(
            root=self.args.data,
            cfg=self.data_cfg,
            splits=split,
            tgt_dict=self.tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            speaker_to_id=self.speaker_to_id,
            multitask=self.multitask_tasks,
        )

    

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        #TODO
        raise NotImplementedError


    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):  
        model.train()
        sample["update_num"] = update_num      
        

        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()

        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
            if self.args.eval_bleu:
                bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
                logging_output["_bleu_sys_len"] = bleu.sys_len
                logging_output["_bleu_ref_len"] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

        return loss, sample_size, logging_output
    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:
            def sum_logs(key):
                if key in logging_outputs[0]:
                    return sum(log[key].cpu().numpy() for log in logging_outputs)
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe_tokenizer is not None:
                s = self.bpe_tokenizer.decode(s)
            if self.pre_tokenizer is not None:
                s = self.pre_tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            )
            hyps.append(hyp)
            refs.append(ref)

        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
        
