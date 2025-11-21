# Hotel Review Classification – Deceptive Opinion Spam Detection

This project implements a machine-learning classifier to detect **deceptive** vs **truthful** hotel reviews, and whether the sentiment is **positive** or **negative**.  
It uses classical NLP (TF-IDF + SVM) to classify each review into one of four categories:

- **TRUTHFULPOSITIVE**
- **TRUTHFULNEGATIVE**
- **DECEPTIVEPOSITIVE**
- **DECEPTIVENEGATIVE**

The project trains on the provided labelled dataset and generates predictions for an unlabeled test set.

---


## Requirements

Create a virtual environment (recommended) and install dependencies:

```bash pip install -r requirements.txt

## Run the code from inside the project root directory:
python src/PROJECT2_NL.py

The script will:
    1)  Load train.txt
    2)  Clean the text minimally
    3)  Transform text with TF-IDF (1–2 grams, English stopwords removed)
    4)  Split into train/test (75/25)
    5)  Train an SVM classifier (RBF kernel or LinearSVC depending on config)
    6)  Evaluate the model + show confusion matrix
    7)  Load the unlabeled test reviews
    8)  Predict their classes
    9)  Save predictions into: Results.txt

