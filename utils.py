import torch
def transform_batch(model,batch_texts,batch_labels,bert_tokenizer,device,vocab,max_length=512):
        batch_input_ids, batch_att_mask,batch_label_ids = [],[],[]
        
        for text, labels in zip(batch_texts,batch_labels):
            assert isinstance(text,list)
            text = ' '.join(text)
            encoded_dict = bert_tokenizer.encode_plus(
                text,
                max_length=max_length,
                padding = 'max_length',
                return_tensors = 'pt',
                truncation = True
            )
            
            batch_input_ids.append(encoded_dict['input_ids'].squeeze(0))
            batch_att_mask.append(encoded_dict['attention_mask'].squeeze(0))
            
            batch_label_ids.append(vocab.tag2id[labels[0]])
            
        batch_input_ids = torch.stack(batch_input_ids)
        batch_att_mask = torch.stack(batch_att_mask)
        batch_label_ids = torch.tensor(batch_label_ids)
        
        batch_input_ids, batch_att_mask,batch_label_ids =\
            batch_input_ids.to(device), batch_att_mask.to(device),batch_label_ids.to(device)
            
        return batch_input_ids, batch_att_mask,batch_label_ids