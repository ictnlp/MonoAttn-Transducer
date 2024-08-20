# Learning Monotonic Attention in Transducer for Streaming Generation


**Files**:

- We mainly provide the following files as plugins into  [``fairseq:920a54``](https://github.com/facebookresearch/fairseq/tree/920a548ca770fb1a951f7f4289b4d3a0c1bc226f) in the [``fs_plugins``](https://github.com/ictnlp/MonoAttn-Transducer/tree/main/fs_plugins) directory.

   ```
   fs_plugins
   ├── agents
   │   ├── attention_transducer_agent.py
   │   ├── monotonic_transducer_agent.py
   |   ├── transducer_agent.py         
   │   └── transducer_agent_v2.py
   ├── criterions
   │   ├── __init__.py
   │   ├── transducer_loss.py             
   │   └── transducer_loss_asr.py                    
   ├── datasets
   │   └── transducer_speech_to_text_dataset.py
   ├── models
   │   ├── transducer
   │   │    ├── __init__.py
   │   │    ├── attention_transducer.py
   │   │    ├── monotonic_transducer.py
   │   │    ├── monotonic_transducer_chunk_diagonal_prior.py
   │   │    ├── monotonic_transducer_chunk_diagonal_prior_only.py
   │   │    ├── monotonic_transducer_diagonal_prior.py
   │   │    ├── transducer.py
   │   │    ├── transducer_config.py
   │   │    └── transducer_loss.py
   │   └── __init__.py
   ├── modules 
   │   ├── attention_transducer_decoder.py
   │   ├── audio_convs.py
   │   ├── audio_encoder.py
   │   ├── monotonic_transducer_decoder.py
   │   ├── monotonic_transformer_layer.py
   │   ├── multihead_attention_patched.py
   │   ├── multihead_attention_relative.py
   │   ├── rand_pos.py
   │   ├── transducer_decoder.py
   │   ├── transducer_monotonic_multihead_attention.py
   │   └── unidirectional_encoder.py
   ├── optim 
   │   ├── __init__.py 
   │   └── radam.py
   ├── scripts 
   │   ├── average_checkpoints.py
   |   ├── prep_mustc_data.py
   │   └── substitute_target.py
   ├── tasks 
   │   ├── __init__.py 
   │   └── transducer_speech_to_text.py
   ├── __init__.py
   └── utils.py
   ```
## Data Preparation
Please refer to [Fairseq's speech-to-text modeling tutorial](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/README.md).

## Training Transformer-Transducer
### ASR Pretraining
We use a batch size of approximating 160k tokens **(GPU number * max_tokens * update_freq == 160k)**.

```bash
main=64
downsample=4
lr=5e-4
warm=4000
dropout=0.1
tokens=8000
language=es

exp=en${language}.asr.cs_${main}.ds_${downsample}.kd.t_t.add.prenorm.amp.adam.lr_${lr}.warm_${warm}.drop_${dropout}.tk_${tokens}.bsz_160k
MUSTC_ROOT=/path_to_your_dataset/mustc/
checkpoint_dir=./checkpoints/$exp

nohup fairseq-train ${MUSTC_ROOT}/en-${language} \
    --amp \
    --config-yaml config_st.yaml --train-subset train_st_distilled --valid-subset dev_st \
    --user-dir fs_plugins \
    --task transducer_speech_to_text --arch t_t \
    --max-source-positions 6000 --max-target-positions 1024 \
    --main-context ${main} --right-context 0 --transducer-downsample ${downsample} \
    --share-decoder-input-output-embed --rand-pos-encoder 300 --encoder-max-relative-position 32 \
    --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion transducer_loss_asr \
    --dropout ${dropout} --weight-decay 0.01 --clip-norm 5.0 \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr ${lr} --lr-scheduler inverse_sqrt \
    --warmup-init-lr '1e-07' --warmup-updates ${warm} \
    --stop-min-lr '1e-09' --max-update 150000 \
    --max-tokens ${tokens} --update-freq 20 --grouped-shuffling \
    --save-dir ${checkpoint_dir} \
    --ddp-backend=legacy_ddp \
    --no-progress-bar --log-format json --log-interval 100 \
    --save-interval-updates 2000 --keep-interval-updates 10 \
    --save-interval 1000 --keep-last-epochs 10 \
    --fixed-validation-seed 7 \
    --skip-invalid-size-inputs-valid-test \
    --validate-interval 1000 --validate-interval-updates 2000 \
    --best-checkpoint-metric rnn_t_loss --keep-best-checkpoints 5 \
    --patience 20 --num-workers 8 \
    --tensorboard-logdir logs_board/$exp >> logs/$exp.txt &
```

### ST Training
We use a batch size of approximating 160k tokens **(GPU number * max_tokens * update_freq == 160k)**.

```bash
main=64
downsample=4
lr=5e-4
warm=4000
dropout=0.1
tokens=8000
language=es
pretrained_path=/path_to_asr_pretrained_checkpoint/avearge.pt


exp=en${language}.s2t.cs_${main}.ds_${downsample}.kd.t_t.add.prenorm.amp.adam.lr_${lr}.warm_${warm}.drop_${dropout}.tk_${tokens}.bsz_160k
MUSTC_ROOT=/path_to_your_dataset/mustc/
checkpoint_dir=./checkpoints/en-${language}/st/$exp

nohup fairseq-train ${MUSTC_ROOT}/en-${language} \
    --load-pretrained-encoder-from ${pretrained_path} \
    --amp \
    --config-yaml config_st.yaml --train-subset train_st_distilled --valid-subset dev_st \
    --user-dir fs_plugins \
    --task transducer_speech_to_text --arch t_t \
    --max-source-positions 6000 --max-target-positions 1024 \
    --main-context ${main} --right-context 0 --transducer-downsample ${downsample} \
    --share-decoder-input-output-embed --rand-pos-encoder 300 --encoder-max-relative-position 32 \
    --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion transducer_loss \
    --dropout ${dropout} --weight-decay 0.01 --clip-norm 5.0 \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr ${lr} --lr-scheduler inverse_sqrt \
    --warmup-init-lr '1e-07' --warmup-updates ${warm} \
    --stop-min-lr '1e-09' --max-update 150000 \
    --max-tokens ${tokens} --update-freq 10 --grouped-shuffling \
    --save-dir ${checkpoint_dir} \
    --ddp-backend=legacy_ddp \
    --no-progress-bar --log-format json --log-interval 100 \
    --save-interval-updates 2000 --keep-interval-updates 10 \
    --save-interval 1000 --keep-last-epochs 10 \
    --fixed-validation-seed 7 \
    --skip-invalid-size-inputs-valid-test \
    --validate-interval 1000 --validate-interval-updates 2000 \
    --best-checkpoint-metric rnn_t_loss --keep-best-checkpoints 5 \
    --patience 20 --num-workers 8 \
    --tensorboard-logdir logs_board/$exp >> logs/$exp.txt &
```

## Training MonoAttn-Transducer

### Offline-Attn Pretraining
We use a batch size of approximating 160k tokens **(GPU number * max_tokens * update_freq == 160k)**.

```bash
main=64
downsample=4
lr=5e-4
warm=4000
dropout=0.1
tokens=8000
language=es
pretrained_path=/path_to_asr_pretrained_checkpoint/avearge.pt  # Use Transformer-Transducer ASR Pretrained Model

exp=en${language}.s2t.cs_${main}.ds_${downsample}.kd.attn_t_t.add.prenorm.amp.adam.lr_${lr}.warm_${warm}.drop_${dropout}.tk_${tokens}.bsz_160k
MUSTC_ROOT=/path_to_your_dataset/mustc/
checkpoint_dir=./checkpoints/en-${language}/st/$exp

nohup fairseq-train ${MUSTC_ROOT}/en-${language} \
    --load-pretrained-encoder-from ${pretrained_path} \
    --amp \
    --config-yaml config_st.yaml --train-subset train_st_distilled --valid-subset dev_st \
    --user-dir fs_plugins \
    --task transducer_speech_to_text --arch attention_t_t \
    --max-source-positions 6000 --max-target-positions 1024 \
    --main-context ${main} --right-context 0 --transducer-downsample ${downsample} \
    --share-decoder-input-output-embed --rand-pos-encoder 300 --encoder-max-relative-position 32 \
    --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion transducer_loss \
    --dropout ${dropout} --weight-decay 0.01 --clip-norm 5.0 \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr ${lr} --lr-scheduler inverse_sqrt \
    --warmup-init-lr '1e-07' --warmup-updates ${warm} \
    --stop-min-lr '1e-09' --max-update 50000 \
    --max-tokens ${tokens} --update-freq 5 --grouped-shuffling \
    --save-dir ${checkpoint_dir} \
    --ddp-backend=legacy_ddp \
    --no-progress-bar --log-format json --log-interval 100 \
    --save-interval-updates 2000 --keep-interval-updates 10 \
    --save-interval 1000 --keep-last-epochs 10 \
    --fixed-validation-seed 7 \
    --skip-invalid-size-inputs-valid-test \
    --validate-interval 1000 --validate-interval-updates 2000 \
    --best-checkpoint-metric rnn_t_loss --keep-best-checkpoints 5 \
    --patience 20 --num-workers 8 --max-tokens-valid 4800 \
    --tensorboard-logdir logs_board/$exp > logs/$exp.txt &
```


### Mono-Attn Training
We use a batch size of approximating 160k tokens **(GPU number * max_tokens * update_freq == 160k)**.

```bash
main=64
downsample=4
lr=5e-4
warm=4000
dropout=0.1
tokens=10000
language=es
                                 
pretrained_path=/path_to_offline_attn_trained_model/average.pt

exp=en${language}.s2t.cs_${main}.ds_${downsample}.kd.mono_t_t_chunk_dia_prior.add.prenorm.amp.adam.lr_${lr}.warm_${warm}.drop_${dropout}.tk_${tokens}.bsz_160k
MUSTC_ROOT=/path_to_your_dataset/mustc/
checkpoint_dir=./checkpoints/en-${language}/st/$exp

nohup fairseq-train ${MUSTC_ROOT}/en-${language} \
    --load-pretrained-encoder-from ${pretrained_path} \
    --load-pretrained-decoder-from ${pretrained_path} \
    --amp \
    --config-yaml config_st.yaml --train-subset train_st_distilled --valid-subset dev_st \
    --user-dir fs_plugins \
    --task transducer_speech_to_text --arch monotonic_t_t_chunk_diagonal_prior \
    --max-source-positions 6000 --max-target-positions 1024 \
    --main-context ${main} --right-context ${main} --transducer-downsample ${downsample} \
    --share-decoder-input-output-embed --rand-pos-encoder 300 --encoder-max-relative-position 32 \
    --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion transducer_loss \
    --dropout ${dropout} --weight-decay 0.01 --clip-norm 5.0 \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr ${lr} --lr-scheduler inverse_sqrt \
    --warmup-init-lr '1e-07' --warmup-updates ${warm} \
    --stop-min-lr '1e-09' --max-update 20000 \
    --max-tokens ${tokens} --update-freq 8 --grouped-shuffling \
    --save-dir ${checkpoint_dir} \
    --ddp-backend=legacy_ddp \
    --no-progress-bar --log-format json --log-interval 100 \
    --save-interval-updates 2000 --keep-interval-updates 20 \
    --save-interval 1000 --keep-last-epochs 10 \
    --fixed-validation-seed 7 \
    --skip-invalid-size-inputs-valid-test \
    --validate-interval 1000 --validate-interval-updates 2000 \
    --best-checkpoint-metric montonic_rnn_t_loss --keep-best-checkpoints 5 \
    --patience 20 --num-workers 8 \
    --tensorboard-logdir logs_board/$exp > logs/$exp.txt &
```

## Inference
### Testing Transformer-Transducer
Use the agent ```transducer_agent_v2```
```bash
LANGUAGE=es
exp=enes.s2t.cs_64.ds_4.kd.t_t.add.prenorm.amp.adam.lr_5e-4.warm_4000.drop_0.1.tk_10000.bsz_160k
ckpt=average_last_5_40000
file=./checkpoints/en-${LANGUAGE}/st/${exp}/${ckpt}.pt
output_dir=./results/en-${LANGUAGE}/st
main_context=64
downsample=4

simuleval \
    --data-bin /dataset/mustc/en-${LANGUAGE} \
    --source /dataset/mustc/en-${LANGUAGE}/data_segment/tst-COMMON.wav_list --target /dataset/mustc/en-${LANGUAGE}/data_segment/tst-COMMON.${LANGUAGE} \
    --model-path $file \
    --config-yaml config_st.yaml \
    --agent ./fs_plugins/agents/transducer_agent_v2.py \
    --transducer-downsample ${downsample} --main-context ${main_context} --right-context ${main_context} \
    --source-segment-size ${main_context}0 \
    --output $output_dir/${exp}_${ckpt} \
    --quality-metrics BLEU  --latency-metrics AL \
    --device gpu
```

## Inference
### Testing MonoAttn-Transducer
Use the agent ```monotonic_transducer_agent```
```bash
LANGUAGE=es
exp=enes.s2t.cs_64.ds_4.kd.mono_t_t.add.prenorm.amp.adam.lr_5e-4.warm_4000.drop_0.1.tk_10000.bsz_160k
ckpt=average_last_5_40000
file=./checkpoints/en-${LANGUAGE}/st/${exp}/${ckpt}.pt
output_dir=./results/en-${LANGUAGE}/st
main_context=64
downsample=4

simuleval \
    --data-bin /dataset/mustc/en-${LANGUAGE} \
    --source /dataset/mustc/en-${LANGUAGE}/data_segment/tst-COMMON.wav_list --target /dataset/mustc/en-${LANGUAGE}/data_segment/tst-COMMON.${LANGUAGE} \
    --model-path $file \
    --config-yaml config_st.yaml \
    --agent ./fs_plugins/agents/monotonic_transducer_agent.py \
    --transducer-downsample ${downsample} --main-context ${main_context} --right-context ${main_context} \
    --source-segment-size ${main_context}0 \
    --output $output_dir/${exp}_${ckpt} \
    --quality-metrics BLEU  --latency-metrics AL \
    --device gpu
```
## Citing

Please kindly cite us if you find our papers or codes useful.

```
```
