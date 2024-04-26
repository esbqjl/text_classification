import numpy as np
import torch
from sklearn.metrics import accuracy_score
import json
from classification.bert_fc.bert_fc_predictor import BertFCPredictor
from classification.bert_fc.bert_fc_trainer import BertFCTrainer

# setup random seed
seed =0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

def setup_data_config(config_path):
    labels_dict={}
    with open(config_path,'r',encoding='utf-8') as f:
        data = json.load(f)
        labels_dict = data['labels'][0]
    return labels_dict

def read_data(data_path,labels_dict):
    '''
    read original data, return titles, labels
    
    ''' 
    texts,labels = [],[]
    with open(data_path,'r',encoding='utf-8') as f:
        print('current_file',data_path)
        
        for i, line in enumerate(f):
            current=json.loads(line)
            sentence = current['text']
            emotion = labels_dict[str(current['label'])]
            text,label = sentence.strip().split(" "),emotion
            texts.append(text),labels.append([label])
            
        print(data_path,'finish')
    return texts, labels
if __name__ == '__main__':
    # read train dev test 
    labels_dict=setup_data_config('./data/config.jsonl')
    train_path, dev_path, test_path = \
        'data/train.jsonl','data/validation.jsonl','data/test.jsonl'
    (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = \
        read_data(train_path,labels_dict),read_data(dev_path,labels_dict),read_data(test_path,labels_dict)

    trainer= BertFCTrainer(
        pretrainerd_model_dir='./model/bert-base-uncased',model_dir='./temp/bertfc',learning_rate=5e-5
    )

    trainer.train(
        train_texts,train_labels,validate_texts = dev_texts, validate_labels = dev_labels, batch_size=64,epoch=8
    )

    predictor= BertFCPredictor(
        pretrained_model_dir='./model/bert-base-uncased',model_dir='./temp/bertfc'
    )

    predict_labels,_,index = predictor.predict(test_texts, batch_size=64)
    # validation
    test_acc = accuracy_score(test_labels, predict_labels)
    print('test acc', test_acc)

