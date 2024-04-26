from peft import get_peft_model, LoraConfig, TaskType
from transformers import BertForSequenceClassification
import torch

import tensorboard
class Config():
    def __init__(self,bert_based_model_dir,target_modules,label_size,lora_r,lora_alpha):
        self.bert_based_model_dir = bert_based_model_dir
        self.label_size = label_size
        self.lora_r=lora_r
        self.lora_alpha = lora_alpha
        self.target_modules= target_modules
        
def get_lora_model(config):
    '''
    config:
        bert_base_model_dir, label_size, drop_out_rate,
    '''
    bert = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=config.label_size,
        output_attentions=False,
        output_hidden_states=False,
    )
    peft_config = LoraConfig(
        task_type = TaskType.SEQ_CLS,
        inference_mode = True,
        r=config.lora_r,
        lora_alpha =config.lora_alpha,
        lora_dropout = 0.1,
        target_modules = config.target_modules
    )
    model = get_peft_model(bert,peft_config)
    # Freeze pretrained model, just train lora part
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.to(config.device)
    return model
if __name__ == "__main__":
    # print out all parameters' name
    DEBUG = False
    if DEBUG:
        config = Config('./model/bert-base-uncased',["query", "key","value"],6,4,16)
        model = get_lora_model(config)
        for k, v in model.named_parameters():
            print(k, v.shape)
        print("*"*20)
        print(model.print_trainable_parameters())