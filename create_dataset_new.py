import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

# 2) Split SENZA stratificazione per evitare errori
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

# 3) Definisci pipeline TF-IDF + oversampling + modello
def make_pipeline(model):
    return ImbPipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
        ('oversample', RandomOverSampler(random_state=42)),
        ('clf', model)
    ])

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

# 4) GridSearchCV con 5 fold stratificato interno
for name, model in models.items():
    print(f"\n### Tuning {name} ###")
    pipe = make_pipeline(model)
    grid = GridSearchCV(
        pipe,
        param_grids[name],
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    
    # Valutazione su test set
    y_pred = grid.predict(X_test)
    pr, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    print(classification_report(y_test, y_pred, zero_division=0))
    
    results[name] = {'precision': pr, 'recall': rec, 'f1_score': f1}
    
    # Salva modello
    joblib.dump(grid.best_estimator_, f'{name}_best.pkl')

# 5) Output metriche finali
pd.DataFrame(results).T.to_csv('hyperparam_metrics.csv')
print("\nMetriche finali:\n", pd.DataFrame(results).T)
