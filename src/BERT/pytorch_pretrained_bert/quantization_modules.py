"""
----------------------------------------
Authors:
- Ikhyun Cho (ikhyuncho@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.autograd import Function
from collections import OrderedDict
#import xnor_cuda

def quantization(input, bits):
    scale_max = 2**(bits-1)-1
    scale_min = -(2**(bits-1))


    pmax = input.max()
    pmin = input.min()
    scale = scale_max - scale_min
    q_scale = pmax - pmin
    
    quantized = torch.round((input - pmin)*(scale / q_scale) + scale_min)
    
#     quantized = torch.floor((input+(2**(bits)-1)) * (scale / q_scale))
    # print(quantized)
    
    qmax = quantized.max()
    qmin = quantized.min()
    scale_q = quantized.max() - quantized.min()
    scale_dq = pmax - pmin
    
    dequantized = (quantized - qmin)*(scale_dq / scale_q) + pmin
    
    return dequantized

def calculate_next_quantization_parts(model_state_dict = dict, current_quantization_step = 0):
    my_dict = {}
    order_dict = {}
    for key in model_state_dict.keys():
        #print(key)
        if 'bias' not in key:
            
            weight_0 = model_state_dict[key]
            weight_00 = model_state_dict[key]
            if current_quantization_step == 0:
                current_step = 8
                next_step = 4
            elif current_quantization_step == 1:
                current_step = 4
                next_step = 1
            else:
                current_step = 1
                next_step = 1
                
            weight_1 = quantization(weight_0, current_step)
            quantized_weight = quantization(weight_0, next_step)
            loss = nn.MSELoss()(weight_0, quantized_weight)
            #print(key+"= "+str(weight_0.size())+"// Quantization error: "+str(loss))
            my_dict.update({key: loss})
    print('='*77)
    print('='*77)
    print('='*77)
    d_ascending = dict(sorted(my_dict.items(), key=lambda item: item[1]))
    count = 0
    print("Checking")
    for key in my_dict.keys():
        print(key,my_dict[key])
#     for key in d_ascending.keys():
#         order_dict.update({key: count})
#         count += 1
#     for key in order_dict.keys():
#         print(key+": "+str(order_dict[key]))