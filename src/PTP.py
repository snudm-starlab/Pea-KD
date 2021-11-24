"""
A module that runs Pretraining with Teacher's Predictions (PTP) on student models. 
The basic structure of the code is analogous to 'src/finetune.py'
"""

import logging
import os
import random
import pickle
import glob
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from envs import PROJECT_FOLDER, HOME_DATA_FOLDER, HOME_OUTPUT_FOLDER

from BERT.pytorch_pretrained_bert.modeling import BertConfig
from BERT.pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from BERT.pytorch_pretrained_bert.tokenization import BertTokenizer

from utils.argument_parser import default_parser, get_predefine_argv, complete_argument
from utils.nli_data_processing import processors, output_modes, init_pretrain_model_PTP, init_pretrain_model_PTP_SPS, get_pretrain_dataloader_PTP
from utils.data_processing import init_model, get_task_dataloader
from utils.modeling import BertForSequenceClassificationEncoder, FCClassifierForSequenceClassification, FullFCClassifierForSequenceClassification
from utils.utils import load_model, count_parameters, eval_model_dataloader_nli, eval_model_dataloader, load_model_finetune
from utils.KD_loss import distillation_loss, patience_loss
from envs import HOME_DATA_FOLDER

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


#########################################################################
# Prepare Parser
##########################################################################
parser = default_parser()
DEBUG = True
logger.info("IN CMD MODE")
args = parser.parse_args()
train_seed_fixed = args.train_seed
saving_criterion_acc_fixed = args.saving_criterion_acc
saving_criterion_loss_fixed = args.saving_criterion_loss
eval_batch_size_fixed = args.eval_batch_size
train_batch_size_fixed = args.train_batch_size
model_type_fixed = args.model_type
save_model_dir_fixed = args.save_model_dir
output_dir_fixed = args.output_dir
load_model_dir_fixed = args.load_model_dir
layer_initialization_fixed = args.layer_initialization
freeze_layer_fixed = args.freeze_layer
fp16_fixed = args.fp16
learning_rate_fixed = args.learning_rate
teacher_prediction_fixed = args.teacher_prediction
num_train_epochs_fixed = args.num_train_epochs
task_name_fixed = args.task
train_type_fixed = args.train_type

if DEBUG:
    logger.info("IN DEBUG MODE")
    argv = get_predefine_argv(args, 'glue', task_name = task_name_fixed, train_type=train_type_fixed, student_layers = args.student_hidden_layers)    
    try:
        args = parser.parse_args(argv)
    except NameError:
        raise ValueError('please uncomment one of option above to start training')
else:
    logger.info("IN CMD MODE")
    args = parser.parse_args()
args.output_dir = output_dir_fixed
args = complete_argument(args, args.output_dir)

if train_seed_fixed is not None:
    args.train_seed = train_seed_fixed
if saving_criterion_acc_fixed is not None:
    args.saving_criterion_acc = saving_criterion_acc_fixed
if saving_criterion_loss_fixed is not None:
    args.saving_criterion_loss = saving_criterion_loss_fixed
if train_batch_size_fixed is not None:
    args.train_batch_size = train_batch_size_fixed    
if eval_batch_size_fixed is not None:
    args.eval_batch_size = eval_batch_size_fixed
if save_model_dir_fixed is not None:
    args.save_model_dir = save_model_dir_fixed
if args.load_model_dir is not None:
    args.encoder_checkpoint = args.load_model_dir
if task_name_fixed is not None:
    args.task_name = task_name_fixed
    args.task = task_name_fixed
if layer_initialization_fixed is not None:
    args.layer_initialization = layer_initialization_fixed
if freeze_layer_fixed is not None:
    args.freeze_layer = freeze_layer_fixed
if fp16_fixed is not None:
    args.fp16 = fp16_fixed
if learning_rate_fixed is not None:
    args.learning_rate = learning_rate_fixed
if teacher_prediction_fixed is not None:
    args.teacher_prediction = teacher_prediction_fixed
if num_train_epochs_fixed is not None:
    args.num_train_epochs = num_train_epochs_fixed
if train_type_fixed is not None:
    args.train_type = train_type_fixed 
    
args.model_type = model_type_fixed
args.raw_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_raw', args.task_name)
args.feat_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_feat', args.task_name)

args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
logger.info('actual batch size on all GPU = %d' % args.train_batch_size)
device, n_gpu = args.device, args.n_gpu

random.seed(args.train_seed)
np.random.seed(args.train_seed)
torch.manual_seed(args.train_seed)
if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.train_seed)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Prepare  Data

teacher_summary = args.teacher_prediction
# We use PTP labels for the training.
train_dataloader, all_label_ids = get_pretrain_dataloader_PTP(task_name = args.task, types = 'train', train_type = args.train_type, teacher_summary = teacher_summary)    
eval_dataloader, eval_label_ids = get_pretrain_dataloader_PTP(task_name = args.task, types = 'dev', train_type ='dontmatter', teacher_summary=teacher_summary)
#test_dataloader, test_label_ids = get_pretrain_dataloader_PTP(task_name = 'MRPC', types = 'test', train_type = 'dontmatter')

logger.info("")
logger.info('='*77)
logger.info("PTP_label.eq(0).sum() = "+str(all_label_ids.eq(0).sum()))
logger.info("PTP_label.eq(1).sum() = "+str(all_label_ids.eq(1).sum()))
logger.info("PTP_label.eq(2).sum() = "+str(all_label_ids.eq(2).sum()))
logger.info("PTP_label.eq(3).sum() = "+str(all_label_ids.eq(3).sum()))
logger.info('='*77)

args.num_train_epochs = 6

if args.task == 'RTE':
    num_train_optimization_steps = int(2490/ args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    log_per_step = 1
elif args.task == 'MRPC':
    num_train_optimization_steps = int(3668/ args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    log_per_step = 1
elif args.task == 'CoLA':
    print(8551)
    num_train_optimization_steps = int(8551/ args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    log_per_step = 5
elif args.task == 'SST-2':
    num_train_optimization_steps = int(67349/ args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    log_per_step = 20
else:
    num_train_optimization_steps = int(104743/ args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    log_per_step = 20    
    
#########################################################################
# Prepare model
#########################################################################
student_config = BertConfig(os.path.join(args.bert_model, 'bert_config.json'))
output_all_layers = True


# if original model
if args.model_type == 'Original':
    student_encoder, student_classifier = init_pretrain_model_PTP(args.task_name, output_all_layers, args.student_hidden_layers, student_config)
    
elif args.model_type == 'SPS_no_shuffle':
    student_encoder, student_classifier = init_pretrain_model_PTP_SPS(args.task_name, output_all_layers, args.student_hidden_layers, student_config, shuffle = False)
elif args.model_type == 'SPS':
    student_encoder, student_classifier = init_pretrain_model_PTP_SPS(args.task_name, output_all_layers, args.student_hidden_layers, student_config, shuffle = True)
    
n_student_layer = len(student_encoder.bert.encoder.layer)
layer_initialization = args.layer_initialization.split(',')
for i in range(len(layer_initialization)):
    layer_initialization[i] = int(layer_initialization[i])
    
#args.encoder_checkpoint = '/home/ikhyuncho23/data/outputs/KD/TinyBERT/pytorch_model.bin'
student_encoder = load_model_finetune(student_encoder, layer_initialization, args.encoder_checkpoint, args, 'student', verbose= True)
logger.info('*' * 77)
student_classifier = load_model(student_classifier, args.cls_checkpoint, args, 'classifier', verbose= True)


n_param_student = count_parameters(student_encoder) + count_parameters(student_classifier)
logger.info('number of layers in student model = %d' % n_student_layer)
logger.info('num parameters in student model are %d and %d' % (count_parameters(student_encoder),  count_parameters(student_classifier)))

#########################################################################
# Prepare optimizer
#########################################################################
if args.do_train:
    param_optimizer = list(student_encoder.named_parameters()) + list(student_classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        logger.info('FP16 activate, use apex FusedAdam')
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        logger.info('FP16 is not activated, use BertAdam')
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)


#########################################################################
# Model Training
#########################################################################
if args.do_train:
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    student_encoder.train()
    student_classifier.train()
    

    log_train = open(os.path.join(args.output_dir, 'train_log.txt'), 'w', buffering=1)
    log_eval = open(os.path.join(args.output_dir, 'eval_log.txt'), 'w', buffering=1)
    print('epoch,global_steps,step,acc,loss,kd_loss,ce_loss,AT_loss', file=log_train)
    print('epoch,acc,loss', file=log_eval)
    
    eval_loss_min = 100
    eval_best_acc = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss, tr_ce_loss, tr_kd_loss, tr_acc_1, tr_acc_2 = 0, 0, 0, 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            student_encoder.train()
            student_classifier.train()
           
            batch = tuple(t.to(device) for t in batch)
            if args.train_type == 'ft':
                train_input_ids, label_ids, train_input_mask, train_segment_ids = batch
            else :
                train_input_ids, label_ids, train_input_mask, train_segment_ids, teacher_pred, teacher_patience= batch
            full_output, pooled_output = student_encoder(train_input_ids, train_segment_ids, train_input_mask)
            logits_pred_student = student_classifier(pooled_output)
            if args.kd_model.lower() == 'kd.cls':
                student_patience = torch.stack(full_output[:-1]).transpose(0,1)
            if args.train_type == 'ft':
                _,_, ce_loss = distillation_loss(logits_pred_student, label_ids, None, T=args.T, alpha=args.alpha)
            else:
                loss_dl, kd_loss, ce_loss = distillation_loss(logits_pred_student, label_ids, teacher_pred, T=args.T, alpha= args.alpha)
            if args.beta > 0:
                pt_loss = args.beta * patience_loss(teacher_patience, student_patience, args.normalize_patience)
                loss = loss_dl + pt_loss
            if args.train_type == 'ft':
                loss = ce_loss
            elif args.train_type == 'kd':
                loss = loss_dl
            else:
                loss = loss_dl + pt_loss
                        
            if n_gpu > 1:
                loss = loss.mean()
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            n_sample = train_input_ids.shape[0]
            tr_loss += loss.item() * n_sample
            
            pred_cls_1 = logits_pred_student.data.max(1)[1]
            tr_acc_1 += pred_cls_1.eq(label_ids).sum().cpu().item()
            nb_tr_examples += n_sample
            nb_tr_steps += 1

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                        
                    
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # We stop the training after n epochs due to overfitting
            if (epoch == 4):
                logger.info("")
                logger.info('='*77)
                logger.info("Validation Loss : "+str(eval_loss_min)+" Validation Accuracy : "+str(eval_best_acc))
                raise ValueError('Skipping the rest epochs due to overfitting')
                
            # We evaluate the model on validation dataset            
            if (global_step % log_per_step == 0) & (epoch > -1) :
                student_encoder.eval()
                student_classifier.eval()
                
                eval_loss, eval_acc_1 = 0, 0
                nb_eval_examples, nb_eval_steps = 0, 0
                for step, batch in enumerate(eval_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    train_input_ids, label_ids, train_input_mask, train_segment_ids= batch
                    with torch.no_grad():
                        _, pooled_output = student_encoder(train_input_ids, train_segment_ids, train_input_mask)
                        logits_pred_student = student_classifier(pooled_output)
                        
                        _,_, ce_loss_ = distillation_loss(logits_pred_student, label_ids, teacher_scores= None, T=args.T, alpha=0)            
                    if n_gpu > 1:
                        ce_loss_ = ce_loss_.mean()
                    
                    n_sample = train_input_ids.shape[0]
                    eval_loss += ce_loss_.item() * n_sample

                    pred_cls_1 = logits_pred_student.data.max(1)[1]
                    eval_acc_1 += pred_cls_1.eq(label_ids).sum().cpu().item()
                    nb_eval_examples += n_sample
                    nb_eval_steps += 1

                    if args.gradient_accumulation_steps > 1:
                        ce_loss_ = ce_loss_ / args.gradient_accumulation_steps
            
                eval_loss = eval_loss/nb_eval_examples
                eval_acc_1 = eval_acc_1/nb_eval_examples                
                
                if eval_acc_1 > eval_best_acc:
                    logger.info("")
                    logger.info('='*77)
                    logger.info("Validation Accuracy improved! "+str(eval_best_acc)+" -> "+str(eval_acc_1))
                    logger.info('='*77)
                    eval_best_acc = eval_acc_1
                    # Save the model if the accuracy is higher than the args.saving_criterion_acc
                    if eval_best_acc > args.saving_criterion_acc:
                        if args.n_gpu > 1:
                            torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, 'PTP' + f'.encoder_acc.pkl'))
                            torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, 'PTP' + f'.cls_acc.pkl'))
                        else:
                            torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, 'PTP' + f'.encoder_acc.pkl'))
                            torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, 'PTP' + f'.cls_acc.pkl'))
                        logger.info("Saving the model...")
                
                if eval_loss < eval_loss_min:
                    logger.info("")
                    logger.info('='*77)
                    logger.info("Validation improved! "+str(eval_loss_min)+" -> "+str(eval_loss))
                    logger.info('='*77)
                    eval_loss_min = eval_loss
                    # Save the model if the loss is lower than the args.saving_criterion_loss
                    if eval_loss < args.saving_criterion_loss:
                        if args.n_gpu > 1:
                            torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, 'PTP'+f'.encoder_loss.pkl'))
                            torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, 'PTP'+f'.cls_loss.pkl'))
                        else:
                            torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, 'PTP'+f'.encoder_loss.pkl'))
                            torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, 'PTP'+f'.cls_loss.pkl'))
                        logger.info("Saving the model...")        

logger.info('*'*77)
logger.info("Validation Loss : "+str(eval_loss_min)+" Validation Accuracy : "+str(eval_best_acc))
logger.info('*'*77)
