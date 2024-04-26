import numpy as np
import torch
from sklearn.metrics import accuracy_score
from pathlib import Path
from peft_predictor import PeftPredictor
from classification.bert_fc.bert_fc_predictor import BertFCPredictor

sentence="This place is haunting for sure, I want to go home"

line = sentence.strip()


text=[line.split(" ")]
print(text)


current_dir = Path(__file__).parent
pretrained_model_dir = current_dir / 'model' / 'bert-base-uncased'

# model_dir = current_dir / 'temp' / 'bertfc'
# predictor= BertFCPredictor(
#     pretrained_model_dir=pretrained_model_dir,model_dir=model_dir
# )

model_dir = current_dir / 'temp' / 'peft'
predictor= PeftPredictor(
    pretrained_model_dir,model_dir,8,8
)



predict_labels,top_probs, top_labels = predictor.predict(text, batch_size=64)
print(predict_labels)
print(top_probs)
print(top_labels)