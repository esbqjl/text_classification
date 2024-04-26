import json
import torch
import math
import os
from sklearn.utils import shuffle
from torch.optim import AdamW
from transformers.utils import logging
import random
from torch.nn import CrossEntropyLoss
from classification.base.base_trainer import BaseTrainer
from classification.bert_fc.bert_fc_model import BertFCModel
from classification.vocab import Vocab
import numpy as np

## manifest the random seed
# seed=0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class BertFCTrainer(BaseTrainer):
    def __init__(self,pretrainerd_model_dir, model_dir, learning_rate = 5e-5,
                 ckpt_name ='pytorch_model.bin',vocab_name="vocab.json"):
        self.pretrained_model_dir = pretrainerd_model_dir
        self.model_dir = model_dir
        self.ckpt_name = ckpt_name
        self.vocab_name = vocab_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.learning_rate = learning_rate
        
        self.batch_size = None
        
        self.vocab = Vocab()
        
    def _build_model(self):
        '''Build up the bert fc model'''
        self.model = BertFCModel(self.pretrained_model_dir,self.vocab.label_size)
        # setup AdamW optimizer
        no_decay = ['bias','LayerNorm.weight']
        optimizer_grouped_parameters=[
            {'params':[p for n,p in self.model.named_parameters() if not any (nd in n for nd in no_decay)],
             'weight_decay':0.01},
            {'params':[p for n,p in self.model.named_parameters() if any (nd in n for nd in no_decay)],
             'weight_decay':0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        self.vocab.set_vocab2id(self.model.get_bert_tokenizer().vocab)
        self.vocab.set_id2vocab({_id:char for char, _id in self.vocab.vocab2id.items()})
        self.vocab.set_unk_vocab_id(self.vocab.vocab2id["[UNK]"])
        self.vocab.set_pad_vocab_id(self.vocab.vocab2id["[PAD]"])
        
        self.model.to(self.device)
        
    def _save_config(self):
        config={
            'vocab_size':self.vocab.vocab_size,
            'label_size':self.vocab.label_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epoch':self.epoch,
            'ckpt_name':self.ckpt_name,
            'vocab_name':self.vocab_name,
            'pretrained_model':os.path.basename(self.pretrained_model_dir)
        }
        with open('{}/train_config.json'.format(self.model_dir),'w') as f:
            f.write(json.dumps(config,indent=4))
    def _transform_batch(self,batch_texts,batch_labels,max_length=512):
        batch_input_ids, batch_att_mask,batch_label_ids = [],[],[]
        
        for text, labels in zip(batch_texts,batch_labels):
            assert isinstance(text,list)
            text = ' '.join(text)
            encoded_dict = self.model.bert_tokenizer.encode_plus(
                text,
                max_length=max_length,
                padding = 'max_length',
                return_tensors = 'pt',
                truncation = True
            )
            
            batch_input_ids.append(encoded_dict['input_ids'].squeeze(0))
            batch_att_mask.append(encoded_dict['attention_mask'].squeeze(0))
            
            batch_label_ids.append(self.vocab.tag2id[labels[0]])
            
        batch_input_ids = torch.stack(batch_input_ids)
        batch_att_mask = torch.stack(batch_att_mask)
        batch_label_ids = torch.tensor(batch_label_ids)
        
        batch_input_ids, batch_att_mask,batch_label_ids =\
            batch_input_ids.to(self.device), batch_att_mask.to(self.device),batch_label_ids.to(self.device)
            
        return batch_input_ids, batch_att_mask,batch_label_ids
    
    def train(self, train_texts, labels, validate_texts, validate_labels, batch_size = 30, epoch=10):
        ''' train
        Args:
            train_text: list[list[str]] training dataset
            labels: list[list[str]] dataset labels
            validate_texts: list[list[str]] validate dataset
            validate_labels: list[list[str]] validate dataset labels
            batch_size: int
            epoch: int
        
        '''
        self.batch_size = batch_size
        self.epoch=epoch
        self.vocab.build_vocab(labels=labels,build_texts=False, with_build_in_tag_id=False) # only build up the labels
        self._build_model()
        self.vocab.save_vocab('{}/{}'.format(self.model_dir,self.vocab_name))
        self._save_config()
        
        best_loss = float('inf')
        loss_buff=[]
        max_loss_num = 10
        step=0
        for epoch in range(epoch):
            for batch_idx in range(math.ceil(len(train_texts)/batch_size)):
                text_batch = train_texts[batch_size * batch_idx:batch_size*(batch_idx+1)]
                labels_batch = labels[batch_size * batch_idx:batch_size * (batch_idx+1)] 
                step=step+1
                self.model.train()
                self.model.zero_grad()
                
                #the training process
                batch_max_len = max([len(text) for text in  text_batch]) + 2
                batch_input_ids, batch_att_mask, batch_label_ids = self._transform_batch(text_batch,
                                                                                         labels_batch,
                                                                                         max_length=batch_max_len)
                
                logits= self.model(batch_input_ids,batch_att_mask)
                
                best_labels = torch.argmax(logits, dim=-1)  
                best_labels = best_labels.to(self.device)
                loss = CrossEntropyLoss(ignore_index=-1)(logits,batch_label_ids)
                loss.backward()
                
                self.optimizer.step()
                
                train_acc = self._get_acc_one_step(best_labels,batch_label_ids)
                
                valid_acc,valid_loss = self.validate(validate_texts,validate_labels,sample_size=batch_size)
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
                # if avg_loss and avg_loss < best_loss:
                #     best_loss = avg_loss
                #     torch.save(self.model.state_dict(),'{}/{}'.format(self.model_dir,self.ckpt_name))
                #     logger.info("model_saved")
        logger.info("finished")
    
    def validate(self,validate_texts,validate_labels,sample_size=100):
        '''use validation dataset set from current model
        Args:
            validate_texts: list[list[str]] or np.array. Validate data
            validate_labels: list[list[str]] or np.array. Validate labels
            sample_size: int. sample size(only use batch size, to prevent validate dataset too large)

        Returns:
            float. validate acc, loss
        ''' 
        self.model.eval()
        
        # randomly sample texts and labels of number of sample_size
        
        batch_texts,batch_labels=[
            return_val[:sample_size] for return_val in shuffle(validate_texts,validate_labels)
        ]
        batch_max_len = max(len(text) for text in batch_texts) + 2
        
        with torch.no_grad():
            batch_input_ids, batch_att_mask, batch_label_ids = self._transform_batch(batch_texts,
                                                                                     batch_labels,
                                                                                     max_length=batch_max_len)
            logits= self.model(batch_input_ids, batch_att_mask,labels=batch_label_ids)
            
            best_labels = torch.argmax(logits, dim=-1)  
            best_labels = best_labels.to(self.device)
            acc = self._get_acc_one_step(best_labels,batch_label_ids)
            loss = CrossEntropyLoss(ignore_index=-1)(logits,batch_label_ids)
            return acc,loss
    def _get_acc_one_step(self,labels_predict_batch,labels_batch):
        acc = (labels_predict_batch==labels_batch).sum().item()/labels_batch.shape[0]
        return float(acc)
                
            
        