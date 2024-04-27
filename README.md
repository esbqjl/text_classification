# Text Classification
The Support Vector Machine (SVM) is a powerful and versatile algorithm used primarily for classification tasks. In the context of text classification,
SVM works by finding a hyperplane in a high-dimensional space that best separates different classes of data.
Mathematical Foundation:
At its core, SVM seeks to solve the following optimization problem:
|||W||^2/2 + C\sum^m_{i=1} ζi
![alt text](https://github.com/esbqjl/text_classification/blob/main/svm_1.jpg)
Here, w is the weight vector normal to the hyperplane,ζi are slack variables representing margin violations, and C is a regularization parameter.

Random Forest:
The Random Forest algorithm is a highly effective and versatile machine learning method primarily used for classification and regression tasks.
When applied to text classification, Random Forest operates by creating a multitude of decision trees during training and outputting the class that is the mode of the classes (classification)
or mean prediction (regression) of the individual trees. Random Forest is ensemble learning, combining multiple decision trees to improve the overall performance.

![alt text](https://github.com/esbqjl/text_classification/blob/main/RF_1.jpg)

In this practise, we are using sklearn library to implement SVM and Randomforest for better scaling the performance in various situation.
## Deep Learning

for Deep learning part(bert) please switch branch to bert_lora

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

The main.py need to use nltk, csv, pickle, sklearn library to run.

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone https://github.com/esbqjl/text_classification.git
   ```
2. Install various packages
   ```sh
   pip install sklearn
   ```
   ```sh
   pip install nltk
   ```
   ```sh
   pip install csv
   ```
   ```sh
   pip install pickle
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Just simply run 
 ```sh
python3 svm_RF.py
 ```
when you need to switch to random forest, uncomment the svm code and comment svm code.


<!-- Contact -->
## Contact

Wenjun - wjz@bu.com

Project Link: [https://github.com/esbqjl/text_classification](https://github.com/esbqjl/text_classification)

<p align="right">(<a href="#readme-top">back to top</a>)</p>





