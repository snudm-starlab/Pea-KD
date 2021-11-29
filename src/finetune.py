"""
The main file used to train student and teacher models. Mainly based on [GitHub repository](https://github.com/intersun/PKD-for-BERT-Model-Compression) 
for [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355).
"""

import logging
import os
import random
import pickle

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import torch.nn as nn
from BERT.pytorch_pretrained_bert.modeling import BertConfig
from BERT.pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from BERT.pytorch_pretrained_bert.tokenization import BertTokenizer
from BERT.pytorch_pretrained_bert.quantization_modules import calculate_next_quantization_parts
from utils.argument_parser import default_parser, get_predefine_argv, complete_argument
from utils.nli_data_processing import processors, output_modes
from utils.data_processing import init_model, get_task_dataloader
from utils.modeling import BertForSequenceClassificationEncoder, FCClassifierForSequenceClassification, FullFCClassifierForSequenceClassification
from utils.utils import load_model, count_parameters, eval_model_dataloader_nli_finetune, eval_model_dataloader, compute_metrics, load_model_finetune, eval_model_dataloader
from utils.KD_loss import distillation_loss, patience_loss
from envs import HOME_DATA_FOLDER
from BERT.pytorch_pretrained_bert.quantization_modules import quantization

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


#########################################################################
# Prepare Parser
#########################################################################

parser = default_parser()
DEBUG = True
logger.info("IN CMD MODE")
args = parser.parse_args()
train_seed_fixed = args.train_seed
saving_criterion_acc_fixed = args.saving_criterion_acc
saving_criterion_loss_fixed = args.saving_criterion_loss
train_batch_size_fixed = args.train_batch_size
eval_batch_size_fixed = args.eval_batch_size
model_type_fixed = args.model_type
save_model_dir_fixed = args.save_model_dir
output_dir_fixed = args.output_dir
load_model_dir_fixed = args.load_model_dir
layer_initialization_fixed = args.layer_initialization
shuffle_initialization_fixed = args.shuffle_initialization
freeze_layer_fixed = args.freeze_layer
fp16_fixed = args.fp16
learning_rate_fixed = args.learning_rate
teacher_prediction_fixed = args.teacher_prediction
num_train_epochs_fixed = args.num_train_epochs

task_name_fixed = args.task
if DEBUG:
    logger.info("IN DEBUG MODE")
    
    argv = get_predefine_argv(args, 'glue', args.task, args.train_type, args.student_hidden_layers)
    
    try:
        args = parser.parse_args(argv)
    except NameError:
        raise ValueError('please uncomment one of option above to start training')
else:
    logger.info("IN CMD MODE")
    args = parser.parse_args()

args.output_dir = output_dir_fixed
if load_model_dir_fixed is not None:
    args.load_model_dir = load_model_dir_fixed
args = complete_argument(args, args.output_dir, args.load_model_dir)

    
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
if shuffle_initialization_fixed is not None:
    args.shuffle_initialization = shuffle_initialization_fixed
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
    
if args.task_name.lower() in ['cola', 'mrpc']:
    args.train_batch_size = 32    

args.model_type = model_type_fixed
args.raw_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_raw', args.task_name)
args.feat_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_feat', args.task_name)

args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
logger.info('actual batch size on all GPU = %d' % args.train_batch_size)
device, n_gpu = args.device, args.n_gpu

###################################################################################################################################
    
random.seed(args.train_seed)
np.random.seed(args.train_seed)
torch.manual_seed(args.train_seed)
if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.train_seed)

    if args.student_hidden_layers == 3:
        args.fc_layer_idx = '3,7'
    elif args.student_hidden_layers == 6:
        args.fc_layer_idx = '1,3,5,7,9'
     
    
logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))
    
    
#########################################################################
# Prepare  Data
#########################################################################

task_name = args.task_name.lower()

if task_name not in processors and 'race' not in task_name:
    raise ValueError("Task not found: %s" % (task_name))

if 'race' in task_name:
    pass
else:
    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

if args.do_train:
    train_sampler = SequentialSampler if DEBUG else RandomSampler
    read_set = 'train'
    if args.teacher_prediction is not None and args.alpha > 0:
        logger.info('loading teacher\'s prediction')
        teacher_predictions = pickle.load(open(args.teacher_prediction, 'rb'))['train'] if args.teacher_prediction is not None else None       
        logger.info('teacher acc = %.2f, teacher loss = %.5f' % (teacher_predictions['acc']*100, teacher_predictions['loss']))
        
        teacher_predictions_ = pickle.load(open(args.teacher_prediction, 'rb'))['dev'] if args.teacher_prediction is not None else None        
        logger.info('teacher acc = %.2f, teacher loss = %.5f' % (teacher_predictions_['acc']*100, teacher_predictions_['loss']))
        
        
        if args.kd_model == 'kd':
            train_examples, train_dataloader, _ = get_task_dataloader(task_name, read_set, tokenizer, args, SequentialSampler,
                                                                      batch_size=args.train_batch_size,
                                                                      knowledge=teacher_predictions['pred_logit'])
        else:
            train_examples, train_dataloader, _ = get_task_dataloader(task_name, read_set, tokenizer, args, SequentialSampler,
                                                                      batch_size=args.train_batch_size,
                                                                      knowledge=teacher_predictions['pred_logit'],
                                                                      extra_knowledge=teacher_predictions['feature_maps'])
    else:
        if args.alpha > 0:
            raise ValueError('please specify teacher\'s prediction file for KD training')
        logger.info('runing simple fine-tuning because teacher\'s prediction is not provided')
        train_examples, train_dataloader, _ = get_task_dataloader(task_name, read_set, tokenizer, args, SequentialSampler,
                                                                  batch_size=args.train_batch_size)
        
    num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    # Run prediction for full data
    eval_examples, eval_dataloader, eval_label_ids = get_task_dataloader(task_name, 'dev', tokenizer, args, SequentialSampler, batch_size=args.eval_batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

#########################################################################
# Prepare model
#########################################################################

student_config = BertConfig(os.path.join(args.bert_model, 'bert_config.json'))
if args.kd_model.lower() in ['kd', 'kd.cls', 'kd.u', 'kd.i']:
    logger.info('using normal Knowledge Distillation')
    output_all_layers = (args.kd_model.lower() in ['kd.cls', 'kd.u', 'kd.i'])
    
    student_encoder, student_classifier = init_model(task_name, output_all_layers, args.student_hidden_layers, student_config)
    
    n_student_layer = len(student_encoder.bert.encoder.layer)
    layer_initialization = args.layer_initialization.split(',')
    for i in range(len(layer_initialization)):
        layer_initialization[i] = int(layer_initialization[i])
    
    shuffle_initialization = args.shuffle_initialization.split(',')
    for i in range(len(shuffle_initialization)):
        if ('False' in shuffle_initialization[i]):
            shuffle_initialization[i] = False
        elif ('True' in shuffle_initialization[i]):
            shuffle_initialization[i] = True

    student_encoder = load_model_finetune(student_encoder, layer_initialization, args.encoder_checkpoint, args, 'student', verbose= True)
    logger.info('*' * 77)
    student_classifier = load_model(student_classifier, args.cls_checkpoint, args, 'classifier', verbose= True)
elif args.kd_model.lower() == 'kd.full':
    logger.info('using FULL Knowledge Distillation')
    layer_idx = [int(i) for i in args.fc_layer_idx.split(',')]
    num_fc_layer = len(layer_idx)
    if args.weights is None or args.weights.lower() in ['none']:
        weights = np.array([1] * (num_fc_layer-1) + [num_fc_layer-1]) / 2 / (num_fc_layer-1)
    else:
        weights = [float(w) for w in args.weights.split(',')]
        weights = np.array(weights) / sum(weights)

    assert len(weights) == num_fc_layer, 'number of weights and number of FC layer must be equal to each other'

    # weights = torch.tensor(np.array([1, 1, 1, 1, 2, 6])/12, dtype=torch.float, device=device, requires_grad=False)
    if args.fp16:
        weights = weights.half()
    student_encoder = BertForSequenceClassificationEncoder(student_config, output_all_encoded_layers=True,
                                                           num_hidden_layers=args.student_hidden_layers,
                                                           fix_pooler=True)
    n_student_layer = len(student_encoder.bert.encoder.layer)
    student_encoder = load_model(student_encoder, args.encoder_checkpoint, args, 'student', verbose=True)
    logger.info('*' * 77)

    student_classifier = FullFCClassifierForSequenceClassification(student_config, num_labels, student_config.hidden_size,
                                                                   student_config.hidden_size, 6)
    student_classifier = load_model(student_classifier, args.cls_checkpoint, args, 'exact', verbose=True)
    assert max(layer_idx) <= n_student_layer - 1, 'selected FC layer idx cannot exceed the number of transformers'
else:
    raise ValueError('%s KD not found, please use kd or kd.full' % args.kd)

n_param_student = count_parameters(student_encoder) + count_parameters(student_classifier)
logger.info('number of layers in student model = %d' % n_student_layer)
logger.info('num parameters in student model are %d and %d' % (count_parameters(student_encoder), count_parameters(student_classifier)))


#########################################################################
# Prepare optimizer
#########################################################################

if task_name == 'rte':
    log_per_step = 1
elif task_name == 'mrpc':
    log_per_step = 1
elif task_name == 'cola':
    log_per_step = 10
elif task_name == 'sst-2':
    log_per_step = 10
else:
    log_per_step = 200 



if args.do_train:
    
    
##############################################################################################################################################    
    print('*'*77)    
        # Determine the layers to freeze
    if args.freeze_layer is not None:    
        list_of_frozen_params = []
        list_of_frozen_params_L1 = []
        for count in range(len(args.freeze_layer)):
            for name, param in student_encoder.named_parameters():   
                if 'bert.encoder.layer.'+str(int(args.freeze_layer[count])-1)+'.' in name:
                    param.requires_grad = False
                    list_of_frozen_params.append(name)
                    list_of_frozen_params_L1.append(torch.mean(torch.abs(param)))
        
        print("Following are the list of params that are frozen")
        for a in range(len(list_of_frozen_params)):
            print(list_of_frozen_params[a])
    else:
        print("No layers are frozen")
        print('*'*77)
################################################################################################################################################    
    
    param_optimizer = list(student_encoder.named_parameters()) + list(student_classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
   
    if args.fp16:
        logger.info('FP16 activate, use apex FusedAdam')
        try:
            from apex.contrib.optimizers import FP16_Optimizer
            from apex.contrib.optimizers import FusedAdam
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
# output_model_file = '{}_nlayer.{}_lr.{}_T.{}.alpha.{}_beta.{}_bs.{}'.format(args.task_name, args.student_hidden_layers,
#                                                                             args.learning_rate,
#                                                                             args.T, args.alpha, args.beta,
#                                                                             args.train_batch_size * args.gradient_accumulation_steps)


print("*"*77)
print("The Shuffle is :"+str(shuffle_initialization))
print("*"*77)
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
    eval_best_acc_and_f1 = 0
    eval_best_f1 = 0 
    loss_acc = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss, tr_ce_loss, tr_kd_loss, tr_acc = 0, 0, 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            student_encoder.train()
            student_classifier.train()
            batch = tuple(t.to(device) for t in batch)
            if args.alpha == 0:
                input_ids, input_mask, segment_ids, label_ids = batch
                teacher_pred, teacher_patience = None, None
            else:
                if args.kd_model == 'kd':
                    input_ids, input_mask, segment_ids, label_ids, teacher_pred = batch
                    teacher_patience = None
                else:
                    input_ids, input_mask, segment_ids, label_ids, teacher_pred, teacher_patience = batch
                    if args.fp16:
                        teacher_patience = teacher_patience.half()
                if args.fp16:
                    teacher_pred = teacher_pred.half()

            full_output, pooled_output = student_encoder(input_ids, segment_ids, input_mask, rand_shuffle = shuffle_initialization)    
            
            if args.kd_model.lower() in['kd', 'kd.cls']:
                logits_pred_student = student_classifier(pooled_output)
                if args.kd_model.lower() == 'kd.cls':
                    student_patience = torch.stack(full_output[:-1]).transpose(0,1)                    
                else:
                    student_patience = None
            elif args.kd_model.lower() == 'kd.full':
                logits_pred_student = student_classifier(full_output, weights, layer_idx)
            else:
                raise ValueError(f'{args.kd_model} not implemented yet')
            
            loss_dl, kd_loss, ce_loss = distillation_loss(logits_pred_student, label_ids, teacher_pred, T=args.T, alpha=args.alpha)
            
            if args.beta > 0:
                if student_patience.shape[0] != input_ids.shape[0]:
                    # For RACE
                    n_layer = student_patience.shape[1]
                    student_patience = student_patience.transpose(0, 1).contiguous().view(n_layer, input_ids.shape[0], -1).transpose(0,1)
                pt_loss = args.beta * patience_loss(teacher_patience, student_patience, args.normalize_patience)
                loss = loss_dl + pt_loss
            else:
                pt_loss = torch.tensor(0.0)
                loss = loss_dl

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            n_sample = input_ids.shape[0]
            tr_loss += loss.item() * n_sample
            if isinstance(kd_loss, float):
                tr_kd_loss += kd_loss * n_sample
            else:
                tr_kd_loss += kd_loss.item() * n_sample
            tr_ce_loss += ce_loss.item() * n_sample
            tr_loss_pt = pt_loss.item() * n_sample

            pred_cls = logits_pred_student.data.max(1)[1]
            tr_acc += pred_cls.eq(label_ids).sum().cpu().item()
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
                
                else:
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps, args.warmup_proportion)
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                        
                if (global_step+1) % 50 ==0:
                    print()
                    print('*'*77)
                    for param_group in optimizer.param_groups:
                        print("Current learning rate is: "+str(param_group['lr'])+'/'+'current global step: '+str(global_step))
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

#             if global_step % args.log_every_step == 0:
#                 print('{},{},{},{},{},{},{},{}'.format(epoch+1, global_step, step, tr_acc / nb_tr_examples,
#                                                        tr_loss / nb_tr_examples, tr_kd_loss / nb_tr_examples,
#                                                        tr_ce_loss / nb_tr_examples, tr_loss_pt / nb_tr_examples),
#                       file=log_train)
            
            
            if global_step % 50 == 0:
                if args.freeze_layer is not None:
                    error = 0
                    L1_list = []
                    print()
                    print('*'*77)
                    print("Checking if the parameters are indeed frozen")
                    for name, param in student_encoder.named_parameters():
                        if name in list_of_frozen_params:
                             L1_list.append(torch.mean(torch.abs(param)))
    #                         print(name+": "+str(torch.mean(torch.abs(param))))
                    for i in range(len(list_of_frozen_params_L1)):
                        if L1_list[i] != list_of_frozen_params_L1[i]:
                            error +=1
                    if error !=0:
                        print("error has occured")
                    else:
                        print("Parameters are well frozen")
                    print('*'*77)    

            if args.num_train_epochs == 10:
                if (epoch == 7):
                    logger.info("*"*77)
                    logger.info("Best Acc: "+str(eval_best_acc)+", Best Loss: "+str(eval_loss_min))
                    if task_name == 'mrpc':
                        logger.info("Best acc and f1: "+str(eval_best_acc_and_f1))
                    logger.info("*"*77)
                    raise ValueError("Skipping the rest.")
                    
            elif args.num_train_epochs == 6:
                if (epoch == 5):
                    logger.info("*"*77)
                    logger.info("Best Acc: "+str(eval_best_acc)+", Best Loss: "+str(eval_loss_min))
                    if task_name == 'mrpc':
                        logger.info("Best acc and f1: "+str(eval_best_acc_and_f1))
                    logger.info("*"*77)
                    raise ValueError("Skipping the rest.")
                    
        #Save a trained model and the associated configuration
            if (global_step % log_per_step == 0) & (epoch > 0): 
                if 'race' in task_name:
                    result = eval_model_dataloader_nli(student_encoder, student_classifier, eval_dataloader, device, False)
                else:
                    test_res = eval_model_dataloader_nli_finetune(args.task_name.lower(), eval_label_ids, student_encoder, student_classifier, eval_dataloader, args.kd_model, num_labels, device, args.weights, args.fc_layer_idx, output_mode, rand_shuffle = shuffle_initialization)
                    
#                 if task_name == 'cola':
#                     print('{},{},{}'.format(epoch+1, test_res['mcc'], test_res['eval_loss']), file=log_eval)
#                 elif task_name == 'mrpc':
#                     print('{},{},{}'.format(epoch+1, test_res['f1'], test_res['eval_loss']), file=log_eval)
#                 else:
#                     print('{},{},{}'.format(epoch+1, test_res['acc'], test_res['eval_loss']), file=log_eval)
                
                
                # Saving checkpoints when the conditions below are met.
                if task_name == 'cola':
                    if test_res['mcc'] > eval_best_acc:
                        logger.info("")
                        logger.info('='*77)
                        logger.info("Validation mcc improved! "+str(eval_best_acc)+" -> "+str(test_res['mcc']))
                        logger.info('='*77)
                        eval_best_acc = test_res['mcc']
                        if eval_best_acc > args.saving_criterion_acc:
                            if args.n_gpu > 1:
                                torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_acc.pkl'))
                                torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_acc.pkl'))
                            else:
                                torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_acc.pkl'))
                                torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_acc.pkl'))
                            logger.info("Saving the model...")                        
                    if test_res['eval_loss']< eval_loss_min:
                        logger.info("")
                        logger.info('='*77)
                        logger.info("Validation Loss improved! "+str(eval_loss_min)+" -> "+str(test_res['eval_loss']))
                        logger.info('='*77)
                        eval_loss_min = test_res['eval_loss']
                        if eval_loss_min < args.saving_criterion_loss:
                            if args.n_gpu > 1:
                                torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_loss.pkl'))
                                torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_loss.pkl'))
                            else:
                                torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_loss.pkl'))
                                torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_loss.pkl'))
                            logger.info("Saving the model...")
                        
                elif task_name == 'mrpc':
                    if test_res['f1'] > eval_best_acc:
                        logger.info("")
                        logger.info('='*77)
                        logger.info("Validation f1 improved! "+str(eval_best_acc)+" -> "+str(test_res['f1']))
                        logger.info('='*77)
                        eval_best_acc = test_res['f1']
                        print("ACC= "+str(test_res['acc']))
                        if eval_best_acc > args.saving_criterion_acc:
                            if args.n_gpu > 1:
                                torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_acc.pkl'))
                                torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_acc.pkl'))
                            else:
                                torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_acc.pkl'))
                                torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_acc.pkl'))
                            logger.info("Saving the model...")
                    if test_res['acc_and_f1'] > eval_best_acc_and_f1:
                        logger.info("")
                        logger.info('='*77)
                        logger.info("Validation acc_and_f1 improved! "+str(eval_best_acc_and_f1)+" -> "+str(test_res['acc_and_f1']))
                        logger.info('='*77)
                        eval_best_acc_and_f1 = test_res['acc_and_f1']
                        if eval_best_acc_and_f1 > args.saving_criterion_acc:
                            if args.n_gpu > 1:
                                torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_acc_and_f1.pkl'))
                                torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_acc_and_f1.pkl'))
                            else:
                                torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_acc_and_f1.pkl'))
                                torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_acc_and_f1.pkl'))
                            logger.info("Saving the model...")                             
                    if test_res['eval_loss']< eval_loss_min:
                        logger.info("")
                        logger.info('='*77)
                        logger.info("Validation Loss improved! "+str(eval_loss_min)+" -> "+str(test_res['eval_loss']))
                        logger.info('='*77)
                        eval_loss_min = test_res['eval_loss']
                        if eval_loss_min < args.saving_criterion_loss:
                            if args.n_gpu > 1:
                                torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_loss.pkl'))
                                torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_loss.pkl'))
                            else:
                                torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_loss.pkl'))
                                torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_loss.pkl'))
                            logger.info("Saving the model...")                    
                else:
                    if test_res['acc'] > eval_best_acc:
                        logger.info("")
                        logger.info('='*77)
                        logger.info("Validation acc improved! "+str(eval_best_acc)+" -> "+str(test_res['acc']))
                        logger.info('='*77)
                        eval_best_acc = test_res['acc']
                        if eval_best_acc > args.saving_criterion_acc:
                            if args.n_gpu > 1:
                                torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_acc.pkl'))
                                torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_acc.pkl'))
                            else:
                                torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_acc.pkl'))
                                torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_acc.pkl'))
                            logger.info("Saving the model...")                        
                    if test_res['eval_loss']< eval_loss_min:
                        logger.info("")
                        logger.info('='*77)
                        logger.info("Validation Loss improved! "+str(eval_loss_min)+" -> "+str(test_res['eval_loss']))
                        logger.info('='*77)
                        eval_loss_min = test_res['eval_loss']
                        if eval_loss_min < args.saving_criterion_loss:
                            if args.n_gpu > 1:
                                torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_loss.pkl'))
                                torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_loss.pkl'))
                            else:
                                torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.encoder_loss.pkl'))
                                torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, 'BERT'+f'.cls_loss.pkl'))
                            logger.info("Saving the model...")                    
                
logger.info("")
logger.info('='*77)
logger.info("Best Acc: "+str(eval_best_acc)+", Best Loss: "+str(eval_loss_min))
logger.info("The seed is : "+str(args.seed))
logger.info('='*77)
