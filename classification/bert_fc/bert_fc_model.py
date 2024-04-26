import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertModel

class BertFCModel(nn.Module):
    
    def __init__(self, bert_base_model_dir, label_size, drop_out_rate = 0.5):
        super(BertFCModel, self).__init__()
        self.label_size = label_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)

        self.bert_model = BertModel.from_pretrained(bert_base_model_dir)
        self.dropout = nn.Dropout(drop_out_rate)
        self.cls_layer = nn.Linear(self.bert_model.config.hidden_size, label_size)
        
    def forward(self,input_ids, attention_mask, token_type_ids = None, Position_ids = None,
                head_mask = None, labels = None):  
        last_hidden_state, pooled_output = self.bert_model(
            input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids,
            position_ids = Position_ids,head_mask=head_mask,inputs_embeds = None, return_dict = False
        )
        cls_outs = self.dropout(pooled_output)
        logits = self.cls_layer(cls_outs)
        return logits

    def get_bert_tokenizer(self):
        return self.bert_tokenizer