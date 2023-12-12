# text_classification
The Support Vector Machine (SVM) is a powerful and versatile algorithm used primarily for classification tasks. In the context of text classification,
SVM works by finding a hyperplane in a high-dimensional space that best separates different classes of data.
Mathematical Foundation:
At its core, SVM seeks to solve the following optimization problem:
|||W||^2/2 + C\sum^m_{i=1} ζi
![alt text]([https://github.com/esbqjl/text_classification/svm_1.jpg](https://github.com/esbqjl/text_classification/blob/main/svm_1.jpg))
Here, w is the weight vector normal to the hyperplane,ζi are slack variables representing margin violations, and C is a regularization parameter.

Random Forest:
The Random Forest algorithm is a highly effective and versatile machine learning method primarily used for classification and regression tasks.
When applied to text classification, Random Forest operates by creating a multitude of decision trees during training and outputting the class that is the mode of the classes (classification)
or mean prediction (regression) of the individual trees. Random Forest is ensemble learning, combining multiple decision trees to improve the overall performance.

![alt text]([https://github.com/esbqjl/text_classification/RF_1.jpg](https://github.com/esbqjl/text_classification/blob/main/RF_1.jpg))
In this practise, we are using sklearn library to implement SVM and Randomforest for better scaling the performance in various situation.

