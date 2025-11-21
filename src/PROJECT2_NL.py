import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC,  LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import string
import joblib
import numpy as np
import os


def save_svm_model(model, model_filename):
    "Function saves the input model"
    joblib.dump(model, model_filename)

def load_svm_model(model_filename):
    "Function loads the input model"
    return joblib.load(model_filename)


def load_data_1(file_path):
    sentences = []
    labels = []

    prefixes = {
        "TRUTHFULPOSITIVE",
        "TRUTHFULNEGATIVE",
        "DECEPTIVEPOSITIVE",
        "DECEPTIVENEGATIVE"
    }

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            label = next((p for p in prefixes if line.startswith(p)), None)
            if label is None:
                continue

            text = line[len(label):].strip()

            if text == "":
                print(f"[WARNING] Empty review at line {idx}, label: {label}. Skipping.")
                continue

            sentences.append(text)
            labels.append(label)

    return sentences, labels




def load_unlabelled(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [line.strip() for line in file if line.strip()]
        return data
    except Exception as e:
        print(f"[ERROR] Could not load unlabelled data: {e}")
        return []



def remove_text_1(text):

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text  

def preprocess_text_1(sentences):
    return [remove_text_1(text) for text in sentences]

def split_data_1(X, y):
    indices = np.arange(len(y))

    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    return X[train_idx], X[test_idx], y_train, y_test, test_idx

def compare_test_results_1(y_pred, y_test, test_indices, raw_sentences):
    errors = {0: [], 1: [], 2: [], 3: []}

    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            true_label = y_test[i]
            sent = raw_sentences[test_indices[i]]
            errors[true_label].append((sent, y_pred[i]))

    return errors


def train_SVM_rbf(X_train, y_train):
    param_grid = {
        'C': [1, 2, 3],
        'gamma': [0.5, 1, 1.5]
    }

    svm = SVC(kernel='rbf')
    grid = GridSearchCV(svm, param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    return grid.best_estimator_


def train_SVM_linear(X_train, y_train):
    param_grid = {'C': [0.5, 1, 2, 5, 10]}

    svm = LinearSVC(class_weight='balanced')
    grid = GridSearchCV(svm, param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    return grid.best_estimator_


def train_RANDOM_FOREST(X, y):
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [None, 20],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5)
    grid.fit(X, y)

    print("Best params:", grid.best_params_)
    return grid.best_estimator_


def predictions_1(X_test, y_test, model):
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    preds = model.predict(X_test)

    if y_test is not None:
        acc = accuracy_score(y_test, preds)
        print("Accuracy:", acc)

        conf = confusion_matrix(y_test, preds)
        labels = [
            "TRUTHFULPOSITIVE",
            "TRUTHFULNEGATIVE",
            "DECEPTIVEPOSITIVE",
            "DECEPTIVENEGATIVE"
        ]

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    return preds


def Write_txt(predict):
    label_mapping = {
        0: "TRUTHFULPOSITIVE",
        1: "TRUTHFULNEGATIVE",
        2: "DECEPTIVEPOSITIVE",
        3: "DECEPTIVENEGATIVE"
    }
    string_predictions = [label_mapping[prediction] for prediction in predict]

    # Write predictions to a text file Results.txt
    
    output_file = "Results.txt"
    with open(output_file, 'w', encoding='utf-8') as output1:
        for prediction in string_predictions:
            output1.write(f"{prediction}\n")
        print(f"Predictions written to {output_file}")
 

def main():
    # -------------------------------------------------------------------
    # 1. DEFINE PATHS
    # -------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    train_file = os.path.join(DATA_DIR, 'train.txt')
    test_file = os.path.join(DATA_DIR, 'test_just_reviews.txt')

    # -------------------------------------------------------------------
    # 2. LOAD AND PREPROCESS TRAINING DATA
    # -------------------------------------------------------------------
    print("[INFO] Loading training data")
    sentences, labels = load_data_1(train_file)

    print("[INFO] Cleaning training text")
    processed_sentences = preprocess_text_1(sentences)

    # map string → integer labels
    class_labels = [
        "TRUTHFULPOSITIVE",
        "TRUTHFULNEGATIVE",
        "DECEPTIVEPOSITIVE",
        "DECEPTIVENEGATIVE"
    ]
    Y = np.array([class_labels.index(label) for label in labels])

    # -------------------------------------------------------------------
    # 3. LOAD AND PREPROCESS UNLABELLED TEST DATA
    # -------------------------------------------------------------------
    print("[INFO] Loading unlabelled test reviews")
    unlabelled_data = load_unlabelled(test_file)

    print("[INFO] Cleaning test text")
    processed_unlabelled = preprocess_text_1(unlabelled_data)

    # -------------------------------------------------------------------
    # 4. CREATE TF-IDF FEATURES
    # -------------------------------------------------------------------
    print("[INFO] Vectorizing text with TF-IDF")
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=2
    )

    X = tfidf_vectorizer.fit_transform(processed_sentences)
    X_test_unlabelled = tfidf_vectorizer.transform(processed_unlabelled)

    # -------------------------------------------------------------------
    # 5. TRAIN/TEST SPLIT
    # -------------------------------------------------------------------
    print("[INFO] Splitting into train/test sets")
    X_train, X_test, y_train, y_test, test_indices = split_data_1(X, Y)

    # -------------------------------------------------------------------
    # 6. TRAIN MODEL
    # -------------------------------------------------------------------
    print("[INFO] Training model")
    USE_SVM = True
    if USE_SVM:
        model = train_SVM_rbf(X_train, y_train)
    else:
        model = train_RANDOM_FOREST(X_train, y_train)

    # -------------------------------------------------------------------
    # 7. EVALUATE ON HELD-OUT TEST SET
    # -------------------------------------------------------------------
    print("[INFO] Evaluating model")
    y_pred = predictions_1(X_test, y_test, model)

    # -------------------------------------------------------------------
    # 8. OPTIONAL – PRINT MISCLASSIFICATIONS
    # -------------------------------------------------------------------
    PRINT_ERRORS = False
    if PRINT_ERRORS:
        print("[INFO] Analyzing misclassifications")
        errors = compare_test_results_1(y_pred, y_test, test_indices, processed_sentences)

        for label_index, label_name in enumerate(class_labels):
            print(f"\n========== {label_name} misclassified ==========")
            for sent, pred in errors[label_index]:
                print("Sentence:", sent)
                print("Predicted:", pred)
                print()

    # -------------------------------------------------------------------
    # 9. PREDICT ON THE UNLABELLED TEST SET
    # -------------------------------------------------------------------
    print("[INFO] Predicting final output")
    final_predictions = predictions_1(X_test_unlabelled, None, model)

    # -------------------------------------------------------------------
    # 10. SAVE RESULTS
    # -------------------------------------------------------------------
    Write_txt(final_predictions)
    print("[INFO] Done.")

main()