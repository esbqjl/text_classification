import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to read JSONL file
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Read datasets
train_data = read_jsonl('./data/train.jsonl')
validation_data = read_jsonl('./data/validation.jsonl')
test_data = read_jsonl('./data/test.jsonl')

# Preprocessing function remains the same
def preprocess_text(text):
    # Tokenization and lowercasing
    tokens = word_tokenize(text.lower())
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    
    return " ".join(filtered_tokens)

# Preprocess and prepare datasets
def prepare_dataset(data):
    texts = [preprocess_text(item['text']) for item in data]
    labels = [item['label'] for item in data]
    return texts, labels

X_train_texts, y_train = prepare_dataset(train_data)
X_val_texts, y_val = prepare_dataset(validation_data)
X_test_texts, y_test = prepare_dataset(test_data)

# Vectorization - fit on training data, transform on all datasets
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_texts)

X_val = vectorizer.transform(X_val_texts)
X_test = vectorizer.transform(X_test_texts)

# Train SVM model
model = svm.SVC(kernel='rbf', C=0.5)
model.fit(X_train, y_train)
# random forest
#model = RandomForestClassifier(n_estimators=100, random_state=42)
#model.fit(X_train, y_train)
# Evaluate on validation data
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))

# Evaluate on test data
y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

def evaluate_and_save(feature, model, X, y, dataset_name, csv_writer):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    # Getting weighted metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    
    csv_writer.writerow([feature, dataset_name, accuracy, precision, recall, f1_score])

# Open a CSV file to write the results
with open('./RF_result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Writing the headers
    # writer.writerow(['kernel', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    writer.writerow(['n_trees', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    # Evaluate on validation data and save
    # kernels=["linear","rbf","sigmoid","poly"]
    n_s= [100,200,300,400]
    # for i in kernels:
    for i in n_s:    
        # model = svm.SVC(kernel=i, C=0.5)
        model = RandomForestClassifier(n_estimators=i, random_state=42)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        evaluate_and_save(i, model, X_val, y_val, 'Validation', writer)

        # Evaluate on test data and save
        evaluate_and_save(' ', model, X_test, y_test, 'Test', writer)
