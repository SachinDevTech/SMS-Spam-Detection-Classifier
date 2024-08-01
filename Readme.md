### Project: SMS Spam Classifier

This project aims to build a robust SMS spam classifier using various machine learning algorithms. The classifier achieves an impressive accuracy of over 97% and a precision of 1. This document details the steps taken to achieve these results, including data preprocessing, model building, evaluation, and deployment.

#### Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Building and Evaluation](#model-building-and-evaluation)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [How to Use](#how-to-use)
8. [Acknowledgements](#acknowledgements)

---

### Introduction

The SMS Spam Classifier is designed to distinguish between legitimate (ham) messages and spam messages. It utilizes a variety of machine learning algorithms to identify patterns in SMS messages and classify them accurately.

### Dataset

The dataset used in this project is the SMS Spam Collection, which contains 5,574 SMS messages labeled as either ham (legitimate) or spam. The dataset can be found [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

### Data Preprocessing

The data preprocessing steps include:

1. **Lowercasing**: Converting all text to lowercase.
2. **Tokenization**: Splitting text into individual words.
3. **Removing non-alphanumeric characters**: Keeping only words and numbers.
4. **Removing stopwords**: Filtering out common words that do not contribute to classification.
5. **Stemming**: Reducing words to their root form.

### Model Building and Evaluation

Several machine learning algorithms were tested to find the best model for this classification task. The performance of each model was evaluated using accuracy and precision metrics.

#### Algorithm Performance

| Algorithm    | Accuracy       | Precision      |
|--------------|----------------|----------------|
| SVC          | 0.9758         | 0.9748         |
| K-Nearest Neighbors (KN) | 0.9052         | 1.0            |
| Naive Bayes (NB) | 0.9710         | 1.0            |
| Decision Tree (DT) | 0.9275         | 0.8119         |
| Logistic Regression (LR) | 0.9584         | 0.9703         |
| Random Forest (RF) | 0.9758         | 0.9829         |
| AdaBoost     | 0.9603         | 0.9292         |
| Bagging Classifier (BgC) | 0.9584         | 0.8682         |
| Extra Trees Classifier (ETC) | 0.9749         | 0.9746         |
| Gradient Boosting Decision Tree (GBDT) | 0.9468         | 0.9192         |
| XGBoost (xgb) | 0.9671         | 0.9262         |


### Results

The Naive Bayes model was chosen for its simplicity and high performance, achieving an accuracy of 97.10% and a precision of 1.0. The voting classifier provided even better results but was not selected for deployment due to its complexity.

### Conclusion

The Naive Bayes classifier is an effective model for SMS spam detection, providing high accuracy and precision. This project demonstrates the importance of text preprocessing and model evaluation in building a reliable machine learning application.

### How to Use

To clone this repository and use the SMS Spam Classifier, follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/SachinDevTech/SMS-Spam-Detection-Classifier.git
   cd SMS-Spam-Detection-Classifier
   ```
2. Run in anaconda, google colab and any other IDE like as VS Code etc...

### Acknowledgements

The dataset used in this project was collected from various sources for SMS spam research. The detailed information about the dataset and its sources can be found [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

By following the steps outlined in this document, you can replicate the results of this project and learn how to build a machine learning-based SMS spam classifier. Feel free to explore the code, experiment with different models, and improve the classifier further.

### Images and Notebook

To embed images and a Jupyter notebook file into the README file, use the following syntax:

#### Embedding Images

![Image Description]()


#### Embedding Jupyter Notebook

You can embed a Jupyter notebook in your README by linking to it. GitHub will render the notebook.

[Open the Notebook](SMS_Spam_Detection.ipynb)

For example:

[Open the Jupyter Notebook](notebooks/SMS_Spam_Detection.ipynb)
```