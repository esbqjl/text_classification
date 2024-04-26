import torch
import json
import math
from peft_model import get_lora_model
from classification.vocab import Vocab
import torch.nn.functional as F
from transformers import BertTokenizer
from Config import Config
import os
import csv
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
class PeftPredictor():
    def __init__(self, pretrained_model_dir, model_dir, lora_r, lora_alpha, vocab_name = 'vocab.json'):
        self.pretrained_mode_dir = pretrained_model_dir
        self.model_dir=model_dir
        self.lora_r = lora_r
        self.lora_alpha=lora_alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = Vocab()
        self._config = Config()
        self._config.add(self._load_config())
        self.vocab.load_vocab('{}/{}'.format(model_dir, vocab_name))
        self._load_model()
        
    def _load_config(self):
        print(os.getcwd())
        with open('{}/train_r{}_alpha{}_config.json'.format(self.model_dir, self.lora_r, self.lora_alpha),'r') as f:
            return json.loads(f.read())
    
    def _load_model(self):
        self.model=get_lora_model(self._config)
        print('{}/{}'.format(self.model_dir,self._config.ckpt_name))
        self.model.load_state_dict(
            torch.load('{}/{}'.format(self.model_dir,self._config.ckpt_name), map_location = self.device)
        )
        self.model.eval()
        
        self.model.to(self.device)
        
        self.vocab.set_unk_vocab_id(self.vocab.vocab2id['[UNK]'])
        self.vocab.set_pad_vocab_id(self.vocab.vocab2id['[PAD]'])
    
    def predict(self,texts,batch_size=64, max_len=512):
        '''
        Args:
            texts: list[list[str]]. to predict the sample, if it's segment, make sure it is segmented by bert's tokenizer
            batch_size: int
            max_len: int the longest sequence length (make sure it remain the same with max_postion_embeding in bert pretrained model)\
            
        Returns:
            list[list[str]] label sequence
        
        '''
        batch_labels = []
        probabilities=[]
        possible_index=[]
        bert_tokenizer =BertTokenizer.from_pretrained(self.pretrained_mode_dir)
        for batch_idx in range(math.ceil(len(texts)/batch_size)):
            text_batch = texts[batch_size * batch_idx : batch_size * (batch_idx+1)]
            
            # longest length of current batch        
            batch_max_len = min(max([len(texts) for texts in text_batch])+2,max_len)
            
            batch_input_ids,batch_att_mask = [],[]
            for text in text_batch:
                assert isinstance(text,list)
                text = ' '.join(text)
                encoded_dict = bert_tokenizer.encode_plus(text, max_length=batch_max_len,padding='max_length',
                                                                        return_tensors='pt',truncation=True)
                batch_input_ids.append(encoded_dict['input_ids'].squeeze(0))
                batch_att_mask.append(encoded_dict['attention_mask'].squeeze(0))
                
            batch_input_ids = torch.stack(batch_input_ids)
            batch_att_mask = torch.stack(batch_att_mask)
            
            batch_input_ids, batch_att_mask = \
                batch_input_ids.to(self.device), batch_att_mask.to(self.device)
            
            with torch.no_grad():
                y_pred = self.model(batch_input_ids, attention_mask=batch_att_mask)
                if not isinstance(y_pred, torch.Tensor):
                        y_pred = y_pred.logits
                
                best_labels = torch.argmax(y_pred, dim=-1)  
                best_labels = best_labels.to(self.device)
                batch_labels.extend([self.vocab.id2tag[label_id.item()] for label_id in best_labels])
                probability=F.softmax(y_pred, dim=-1)
                top_probs, top_labels = torch.topk(probability, 5, dim=-1)
                for i in range(len(top_labels)):
                   
                    possible_index.append([self.vocab.id2tag[j.item()]  for j in top_labels[i]])
        return batch_labels,top_probs, possible_index
    
    def evaluate(self, predicted_labels, true_labels):
        # Convert labels to indices for metrics calculation
        
        true_indices = true_labels
        pred_indices = [label for label in predicted_labels]
        print(len(true_indices))
        print(len(pred_indices))
        
        # Calculating metrics
        cm = confusion_matrix(true_indices, pred_indices)
        accuracy = accuracy_score(true_indices, pred_indices)
        precision = precision_score(true_indices, pred_indices, average='macro')
        recall = recall_score(true_indices, pred_indices, average='macro')
        f1 = f1_score(true_indices, pred_indices, average='macro')

        # Writing to CSV
        file_exists = os.path.exists('lora_result.csv')
        with open('lora_result.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(['Metric', 'Value'])
            if not file_exists:
                writer.writerow(['lora_r', 'lora_alpha', 'Accuracy', 'Precision', 'F1 Score', 'Recall'])
            writer.writerow([self.lora_r, self.lora_alpha, accuracy, precision, f1, recall])
        # Optionally save confusion matrix
        # cm_df = pd.DataFrame(cm, index=[label for label in label_indices], columns=[label for label in label_indices])
        # cm_df.to_csv('confusion_matrix.csv')

        return cm, accuracy, precision, recall, f1