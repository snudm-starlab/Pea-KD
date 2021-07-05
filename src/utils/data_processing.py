"""
File used to initialize models and make dataloaders. Mainly based on [GitHub repository](https://github.com/intersun/PKD-for-BERT-Model-Compression) for [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355).
"""
from utils.nli_data_processing import init_glue_model, get_glue_task_dataloader, get_glue_task_dataloader_pretrain5, init_glue_model_ER
from utils.race_data_processing import init_race_model, get_race_task_dataloader


def init_model(task_name, output_all_layers, num_hidden_layers, config):
    if 'race' in task_name.lower():
        return init_race_model(task_name, output_all_layers, num_hidden_layers, config)
    else:
        return init_glue_model(task_name, output_all_layers, num_hidden_layers, config)

def init_model_ER(task_name, output_all_layers, num_hidden_layers, config, shuffle=False):
    if 'race' in task_name.lower():
        return init_race_model(task_name, output_all_layers, num_hidden_layers, config)
    else:
        return init_glue_model_ER(task_name, output_all_layers, num_hidden_layers, config)
    
def get_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size=None, knowledge=None, extra_knowledge=None):
    if 'race' in task_name.lower():
        return get_race_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)
    else:
        return get_glue_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)

def get_task_dataloader_pretrain(task_name, set_name, tokenizer, args, sampler, batch_size=None, knowledge=None, extra_knowledge=None, p5_label = None):
    if 'race' in task_name.lower():
        return get_race_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)
    else:
        return get_glue_task_dataloader_pretrain5(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge, p5_label = p5_label)
