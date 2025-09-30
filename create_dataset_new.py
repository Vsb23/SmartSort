import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
import joblib

# 1) Carica dataset
df = pd.read_csv('training_set_single_category_85percent.csv')
df['clean_text'] = df['clean_text'].fillna('')
df = df.dropna(subset=['category']).reset_index(drop=True)

X = df['clean_text']
y = df['category']

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Definisci pipeline TF-IDF + sampling + modello
def make_pipeline(model):
    return ImbPipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
        ('oversample', RandomOverSampler(random_state=42)),
        ('clf', model)
    ])

# 4) Specifiche griglie di ricerca
param_grids = {
    'LogisticRegression': {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l2'],
        'clf__solver': ['saga']
    },
    'RandomForest': {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 30],
        'clf__min_samples_leaf': [1, 3, 5]
    },
    'SVM': {
        'clf__C': [0.1, 1, 10],
        'clf__gamma': ['scale', 'auto'],
        'clf__kernel': ['rbf']
    }
}

models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'SVM': SVC(class_weight='balanced', probability=True)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5) GridSearchCV per ogni modello
for name, model in models.items():
    print(f"\n### Tuning {name} ###")
    pipe = make_pipeline(model)
    grid = GridSearchCV(
        pipe,
        param_grids[name],
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    # Valuta su test
    y_pred = grid.predict(X_test)
    pr, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    print(classification_report(y_test, y_pred, zero_division=0))
    results[name] = {'precision': pr, 'recall': rec, 'f1_score': f1}
    # Salva modello e vettorizzatore incorporato
    joblib.dump(grid.best_estimator_, f'{name}_best.pkl')

# Report finale
pd.DataFrame(results).T.to_csv('hyperparam_metrics.csv')
print("\nMetriche finali:\n", pd.DataFrame(results).T)

# --- Fine-tuning con Sentence-Transformers BERT ---
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample

# Prepara esempi
train_examples = [
    InputExample(texts=[text], label=float(label_idx)) 
    for text, label_idx in zip(X_train, y_train.factorize()[0])
]

model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.SoftmaxLoss(
    model=model, 
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=len(y.unique())
)

# Fine-tuning
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='bert_finetuned'
)

print("Fine-tuning completato. Modello salvato in 'bert_finetuned'.")
