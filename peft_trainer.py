
import random 
import numpy as np
import torch
from torch.optim import AdamW
from peft_model import get_lora_model, Config
from sklearn.utils import shuffle
from train import setup_data_config, read_data
import math
from classification.vocab import Vocab
from utils import transform_batch
from transformers.utils import logging
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer
from peft_predictor import PeftPredictor
import json
from Config import Config
import os
# manifest the random seed
seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
logging.set_verbosity_info()
logger = logging.get_logger("transformers")


        
def validate(model,validate_texts,validate_labels,bert_tokenizer,device,vocab,sample_size=100):
    '''use validation dataset set from current model
    Args:
        validate_texts: list[list[str]] or np.array. Validate data
        validate_labels: list[list[str]] or np.array. Validate labels
        sample_size: int. sample size(only use batch size, to prevent validate dataset too large)

    Returns:
        float. validate acc, loss
    ''' 
    model.eval()
    
    # randomly sample texts and labels of number of sample_size
    
    batch_texts,batch_labels=[
        return_val[:sample_size] for return_val in shuffle(validate_texts,validate_labels)
    ]
    batch_max_len = max(len(text) for text in batch_texts) + 2
    
    with torch.no_grad():
        batch_input_ids, batch_att_mask, batch_label_ids = transform_batch(model,
                                                                        batch_texts,
                                                                        batch_labels,
                                                                        bert_tokenizer,
                                                                        device,
                                                                        vocab,
                                                                        max_length=batch_max_len)
        y_pred = model(batch_input_ids, attention_mask=batch_att_mask)
        if not isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.logits
        best_labels = torch.argmax(y_pred, dim=-1)
        loss = CrossEntropyLoss(ignore_index=-1)(y_pred,batch_label_ids)
        acc = _get_acc_one_step( best_labels,batch_label_ids)
         
        return acc,loss
def _get_acc_one_step(labels_predict_batch,labels_batch):
    
    acc = (labels_predict_batch==labels_batch).sum().item()/labels_batch.shape[0]
    return float(acc)

def save_config(config):
    
    _config={
        'label_size':config.label_size,
        'model_dir' : config.model_dir,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'epoch':config.epoch,
        'ckpt_name':config.ckpt_name,
        'vocab_name':config.vocab_name,
        'bert_based_model_dir':os.path.basename(config.bert_based_model_dir),
        'lora_r' : config.lora_r,
        'lora_alpha': config.lora_alpha,
        'target_modules':config.target_modules,
        'device':str(config.device)
    }
    with open('{}/train_r{}_alpha{}_config.json'.format(config.model_dir, config.lora_r, config.lora_alpha),'w') as f:
        f.write(json.dumps(_config,indent=4))
        
def trainer(config,vocab,bert_tokenizer):
    

    

    model = get_lora_model(config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    batch_size = 64
    config.batch_size = batch_size
    epoch=8
    config.epoch = epoch
    best_loss = float('inf')
    loss_buff=[]
    max_loss_num = 10
    step=0

    bert_based_model_dir,model_dir=config.bert_based_model_dir, config.model_dir

    ckpt_name='pytorch_model_r{}_alpha{}.bin'.format(config.lora_r, config.lora_alpha)
    config.ckpt_name = ckpt_name
    # save_config(config)
    for epoch in range(epoch):
        for batch_idx in range(math.ceil(len(train_texts)/batch_size)):
            text_batch = train_texts[batch_size * batch_idx:batch_size*(batch_idx+1)]
            labels_batch = train_labels[batch_size * batch_idx:batch_size * (batch_idx+1)] 
            step=step+1
            model.train()
            model.zero_grad()
            
            #the training process
            batch_max_len = max([len(text) for text in  text_batch]) + 2
            batch_input_ids, batch_att_mask, batch_label_ids = transform_batch(model,
                                                                            text_batch,
                                                                            labels_batch,
                                                                            bert_tokenizer=bert_tokenizer,
                                                                            device=device,
                                                                            vocab=vocab,
                                                                            max_length=batch_max_len)
            
            y_pred= model(batch_input_ids,attention_mask=batch_att_mask)
            if not isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.logits
            
            best_labels = torch.argmax(y_pred, dim=-1)
            loss = CrossEntropyLoss(ignore_index=-1)(y_pred,batch_label_ids)
            loss.backward()
            optimizer.step()

            train_acc = _get_acc_one_step(best_labels,batch_label_ids)
            valid_acc,valid_loss = validate(model,dev_texts,dev_labels,bert_tokenizer,device,vocab,sample_size=batch_size)
            loss_buff.append(valid_loss)
            if len(loss_buff)>max_loss_num:
                loss_buff = loss_buff[1:]
            avg_loss = sum(loss_buff)/len(loss_buff) if len(loss_buff)==max_loss_num else None
            
            logger.info(
                'epoch %d, step %d, train loss %.4f, train acc %.4f, valid acc %.4f, last %d avg valid loss %s' % (
                    epoch, step, loss, train_acc, valid_acc, max_loss_num,
                    '%.4f' % avg_loss if avg_loss else avg_loss
                )
            )
            if avg_loss and avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(),'{}/{}'.format(model_dir,ckpt_name))
                logger.info("model_saved")
    logger.info("finished")
    

if __name__ == "__main__":
    lora_r_list = [32]
    lora_alpha_list = [64]
    labels_dict=setup_data_config('./data/config.jsonl')
    train_path, dev_path, test_path = \
        'data/train.jsonl','data/validation.jsonl','data/test.jsonl'
    (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = \
        read_data(train_path,labels_dict),read_data(dev_path,labels_dict),read_data(test_path,labels_dict)
        
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("here",device)
    config = Config(bert_based_model_dir='./model/bert-base-uncased',model_dir='./temp/peft',learning_rate=1e-4,device = device)
    
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained(config.bert_based_model_dir)
    vocab.build_vocab(labels=train_labels,build_texts=False, with_build_in_tag_id=False)
    vocab.set_vocab2id(bert_tokenizer.vocab)
    vocab.set_id2vocab({_id:char for char, _id in vocab.vocab2id.items()})
    vocab.set_unk_vocab_id(vocab.vocab2id["[UNK]"])
    vocab.set_pad_vocab_id(vocab.vocab2id["[PAD]"])
    vocab.save_vocab('{}/{}'.format(config.model_dir,config.vocab_name))
    


    
    for lora_r in lora_r_list:
        for lora_alpha in lora_alpha_list:
            
            logger.info("start training lora model")
            logger.info("lora_r: {}, lora_alpha: {}".format(lora_r, lora_alpha))
            
            config.lora_r = lora_r
            config.lora_alpha = lora_alpha
            trainer(config,vocab,bert_tokenizer)
            
            predictor= PeftPredictor(
                config.bert_based_model_dir,config.model_dir,config.lora_r,config.lora_alpha
            )
            predict_labels,top_probs, top_labels = predictor.predict(test_texts, batch_size=64)
            predictor.evaluate(predict_labels,test_labels)
            