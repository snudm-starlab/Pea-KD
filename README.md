# Pea-KD 
This project is a PyTorch implementation of Parameter-efficient and accurate Knowledge Distillation on BERT. This paper proposes a novel approach that improves Knowledge Distillation performance. This package is especially for BERT model.  

## Overview
#### Brief Explanation of the paper. 
Two main ideas proposed in the paper. Shuffled Parameter Sharing (SPS) and Pretraining with Teacher's Predictions (PTP). 

1) SPS 

- step1 : Paired Parameter Sharing. 
We first double the layers of the student model. Then, we share the parameters between the bottom half and the upper half. 
By this way, the model has twice the number of layers and thus can have more 'effective' model complexity while having the same number of actual parameters. 

- step2 : Shuffling. 
In addition to step1, we shuffle the Query and Key parameters between the shared pairs in order to further increase the 'effective' model complexity. 
By this shuffling process, the parameter-shared pairs can have higher model complexity and thus better representation power. 
We will call this architecture the SPS model. 

2) PTP 

- We pretrain the student model with new artificial labels (PTP labels). The labels are assigned as follows.

``` Unicode
PTP labels 
  ├── 'Confidently Correct' = teacher model's prediction is correct & confidence > t 
  ├── 'Unconfidently Correct' = teacher model's prediction is correct & confidence <= t 
  ├── 'Confidently Wrong' = teacher model's prediction is wrong & confidence > t 
  └── 'Unconfidently Wrong' = teacher model's prediction is wrong & confidence <= t
  t = hyperparameter : depends on the downstream task and the teacher model. e.g.) t = 0.95 for MRPC, t = 0.8 for RTE.
```  
#### Baseline Codes
This repository is based on the [GitHub repository](https://github.com/intersun/PKD-for-BERT-Model-Compression) for [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355). All source files are from the repository if not mentioned otherwise. The main scripts that actually run tasks are the following two files, and they have been modified from the original files in the original repository:
- 'finetune.py', 'PTP.py' - based on 'NLI_KD_training.py' in the original repository.
- 'save_teacher_outputs.py - based on 'run_glue_benchmark.py' in the original repository.

``` Unicode
PeaKD
  │
  ├──  src        
  │     ├── BERT
  │     │    └── pytorch_pretrained_bert: BERT sturcture files
  │     ├── data
  │     │    ├── data_raw
  │     │    │     ├── glue_data: task dataset
  │     │    │     └── download_glue_data.py
  │     │    ├── models
  │     │    │     └── bert_base_uncased: ckpt
  │     │    └── outputs
  │     │           └── save teacher model prediction & trained student model.
  │     ├── utils : The overall utils. 
  │     ├── envs.py: save directory paths for several usage.
  │     ├── save_teacher_outputs.py : save teacher prediction. Used for PTP, KD, PKD e.t.c. 
  │     ├── PTP.py : pretrain the student model with PTP. 
  │     └── finetune.py: comprehensive training file for teacher and student models.
  │
  ├── preprocess.sh: downloads GLUE datasets.
  ├── Makefile: Makefile used for demo.
  ├── Developers_Guide.docx
  ├── requirements.txt: run this file to download required environments.
  ├── LICENSE
  └── README.md
```

#### Data description
- GLUE datasets

* Note that: 
    * GLUE datasets consists of CoLA, diagnostic, MNLI, MRPC, QNLI, QQP, RTE, SNLI, SST-2, STS-B, WNLI
    * You can download GLUE datasets by running bash 'preprocess.sh'.


## Install 

#### Environment 
* Ubuntu
* CUDA 10.0
* Pytorch 1.4 
* numpy
* torch
* Tensorly
* tqdm
* pandas
* apex

## Dependence Install
```
cd PeaKD
pip install -r requirements.txt
```

# Getting Started

## Preprocess
Download GLUE datasets by running script:
```
bash preprocess.sh
```
You must download your own pretrained BERT model at 'src/data/models/pretrained/bert-base-uncased'. 
Refer to 'src/BERT/pytorch_pretrained_bert/modeling.py' line 43~51.

## Demo 
you can run the demo version.
```
make
```

## Run your own training  
* We provide an example how to run the codes. We use task: 'MRPC', teacher layer: 12, and student layer: 3 as an example.
* Before starting, we need to specify a few things.
    * task: one of the GLUE datasets
    * train_type: one of the followings - ft, kd, pkd 
    * model_type: one of the followings - Original, SPS
    * student_hidden_layers: the number of student layers
    * train_seed: the train seed to use. If default -> random 
    * saving_criterion_acc: if the model's val accuracy is above this value, we save the model.
    * saving_criterion_loss: if the model's val loss is below this value, we save the model.
    * load_model_dir: specify a directory of the checkpoint if you want to load one.
    * output_dir: specify a directory where the outputs will be written and saved.
    
* First, We begin with finetuning the teacher model
    ```
    run script
    python src/finetune.py \
    --task 'MRPC' \
    --train_type 'ft' \
    --model_type 'Original' \
    --student_hidden_layers 12 \
    --saving_criterion_acc 1.0 \
    --saving_criterion_loss 0 .6 \
    --output_dir 'run-1'
    ```
    The trained model will be saved in 'src/data/outputs/KD/{task}/teacher_12layer/'

* To use the teacher model's predictions for PTP, KD, and PKD run script:
    ```
    python src/save_teacher_outputs.py
    ```
    The teacher predictions will be saved in 'src/data/outputs/KD/{task}/{task}_normal_kd_teacher_12layer_result_summary.pkl'
    or 'src/data/outputs/KD/{task}/{task}_patient_kd_teacher_12layer_result_summary.pkl'

* To apply PTP to the student model, run script:
    ```
    run script:
    python src/PTP.py \
    --task 'MRPC' \
    --train_type 'ft' \
    --model_type 'SPS' \
    --student_hidden_layer 3 \
    --saving_criterion_acc 0.8 \
    --output_dir 'run-1'
    ```
    The pretrained student model will be saved in 'src/data/outputs/KD/{task}/teacher_12layer/'. 
    you may specify the hyperparameter 't' in src/utils/nli_data_processing.py line 713~.
* When PTP is done, we can finally finetune the student model by running script:
    ```
    python src/finetune.py \
    --task 'MRPC' \
    --train_type 'pkd' \
    --model_type 'SPS' \
    --student_hidden_layers 3 \
    --saving_criterion_acc 1.0 \
    --saving_criterion_loss 0.6 \
    --load_model_dir 'run-1/PTP.encoder_loss.pkl' \
    --output_dir 'run-1/final_results'
    ```
