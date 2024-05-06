# Email/SMS-spam-detection-system

## Introduction:

Spam detection systems play a crucial role in safeguarding our digital communication channels from unwanted, fraudulent, or malicious content. This SMS/Email Spam Detection Web Application is designed to help users identify and filter out spam messages, whether they arrive via text messages or emails.

## Why Spam Detection?

In an era where digital communication is essential, the volume of unsolicited and potentially harmful messages has surged. Spam emails and SMS messages can range from annoying marketing promotions to phishing attempts and malware delivery. An effective spam detection system is essential for recognizing and filtering out these spam messages.

## Technologies uses:
- Python, pandas and numpy
- Streamlit
- MultiNominal Naive bayes machine learning algorithm
- nltk library
- sci-kit learn, ect

## Evaluation Matrix
* Accuracy: <img src="https://latex.codecogs.com/svg.latex?\frac{TP&plus;TN}{TP&plus;TN&plus;FP&plus;FN}" title="\frac{TP+TN}{TP+TN+FP+FN}" />
* Recall : <img src="https://latex.codecogs.com/svg.latex?\frac{TP}{TP&plus;FN}" title="\frac{TP}{TP+FN}" />
* Precision: <img src="https://latex.codecogs.com/svg.latex?\frac{TP}{TP&plus;FP}" title="\frac{TP}{TP+FP}" />
* F1-Measure: <img src="https://latex.codecogs.com/svg.latex?\frac{2*Recall*Precision}{Recall&plus;Precision}" title="\frac{2*Recall*Precision}{Recall+Precision}" />


## Confusion Matrix

Considering SPAM as a positive class and HAM as the negative class:

|                  | SPAM (Predicted)   | HAM (Predicted)   |
|------------------|--------------------|-------------------|
| SPAM (Actual)    |       TP = 336     |       FN = 64     |
| HAM (Actual)     |       FP =   6     |       TN = 394    |

Accuracy:  0.9125

Precision: 0.9824561403508771

Recall:    0.84

f1-measure: 0.9056603773584906
