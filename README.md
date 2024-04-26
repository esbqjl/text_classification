# Emotion Classification using Bert
<img src="https://github.com/esbqjl/text_classification/blob/bert_lora/image/Bert.jpg" alt="drawing" width="600" height="400"/>

## What Bert is?

"As we learned what a Transformer is and how we might train the Transformer model, we notice that it is a great tool to make a computer understand human language. However, the Transformer was originally designed as a model to translate one language to another. If we repurpose it for a different task, we would likely need to retrain the whole model from scratch. Given the time it takes to train a Transformer model is enormous, we would like to have a solution that enables us to readily reuse the trained Transformer for many different tasks. BERT is such a model. It is an extension of the encoder part of a Transformer."
from https://machinelearningmastery.com/a-brief-introduction-to-bert/

<img src="https://github.com/esbqjl/text_classification/blob/bert_lora/image/BERT_Overall.jpg" alt="drawing" width="1000" height="450"/>

### Pre-training and Fine-tuning
BERT is designed to be pre-trained on a large corpus of text in an unsupervised manner using two strategies:

Masked Language Model (MLM): Randomly masking some of the words in the input and predicting them based only on their context.
Next Sentence Prediction (NSP): Given pairs of sentences as input, the model predicts if the second sentence logically follows the first one.
After pre-training, BERT can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering, sentiment analysis, and language inference.

### Transfer Learning
BERT exemplifies the concept of transfer learning in NLP. The idea is to take a model pre-trained on a large dataset and fine-tune it on a smaller, task-specific dataset. This approach allows BERT to perform well on tasks even when there is relatively little labeled training data available.


## goal of this project

"Text classification is a fundamental task in natural language processing, involving the categorization of textual data into predefined classes. This study aims to investigate the efficacy of various machine learning and deep learning methodologies in text classification. Traditional machine learning methods, such as Logistic Regression and SVM, have been widely used in this domain due to their simplicity and efficiency. However, with the advent of deep learning, more sophisticated approaches like BERT embeddings, LoRA fine-tuning, LSTM, and CNN have emerged, offering enhanced feature extraction and learning capabilities. Our experiment leverages both traditional and deep learning methods to develop a comprehensive text classification model. The traditional approaches provide a baseline for performance comparison, while the deep learning techniques, particularly those based on BERT, are hypothesized to offer superior performance due to their advanced contextual understanding and adaptability. The objective of this analysis is to evaluate these methodologies on a standardized dataset, aiming to reveal the most effective strategies for text classification in the current technological landscape. In this practise, we are using sklearn library to implement SVM and Randomforest for better scaling the performance in various situation." from our papers, see link below.

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

see requirements.txt and setup.py

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone https://github.com/esbqjl/text_classification.git
   ```
2. Switch branch to bert_lora
   ```sh
   git switch bert_lora
   ```
3. Install various packages
   ```sh
   pip install -e .
   ```
4. model and datasets
   you need to download model from https://huggingface.co/google-bert/bert-base-uncased/
   
   you need to download datasets from https://huggingface.co/datasets/dair-ai/emotion/
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Train emotion classifier using pretrained model 
 ```sh
   python3 train.py
 ```
you can run the predictor using predict.py
 ```sh
   python3 predict.py
 ```
Fine tune Bert using lora
 ```sh
   python3 peft_trainer.py
 ```
you can also run this predictor using predict.py but you need to switch the predictor, see the code for more information.
```sh
   python3 predict.py
 ```


<!-- Contact -->
## Contact

Wenjun - wjz@bu.com

Project Link: [https://github.com/esbqjl/text_classification](https://github.com/esbqjl/text_classification)

## Useful Link 
Paper Link : [https://github.com/esbqjl/text_classification/blob/bert_lora/Project__Final_Report.pdf]

Transformers: [https://github.com/huggingface/transformers]

Bert: [https://github.com/codertimo/BERT-pytorch]

Lora: [https://github.com/microsoft/LoRA]

Personal Website + project demonstration: [https://www.happychamber.xyz/deep_learning]
<p align="right">(<a href="#readme-top">back to top</a>)</p>





