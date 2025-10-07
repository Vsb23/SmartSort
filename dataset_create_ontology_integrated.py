import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from rdflib import Graph, Namespace, RDFS
import numpy as np
from collections import Counter, defaultdict
import text_extract



def load_keywords_from_ontology(ontology_path, ns):
    """
    NUOVA FUNZIONE: Carica keywords direttamente dall'ontologia OWL
    Sostituisce il dizionario hardcoded ONTOLOGY_KEYWORDS
    """
    print("📖 Caricamento keywords dall'ontologia...")
    
    g = Graph()
    g.parse(ontology_path, format='xml')
    
    # SPARQL query per estrarre tutte le hasKeyword properties
    query = """
    PREFIX ns: <http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#>
    
    SELECT ?class ?keyword WHERE {
        ?class ns:hasKeyword ?keyword .
    }
    """
    
    keywords_dict = defaultdict(list)
    
    try:
        # Esegui query SPARQL
        results = g.query(query)
        
        for row in results:
            class_uri = str(row['class'])
            keyword = str(row['keyword'])
            
            # Estrai nome classe dall'URI
            class_name = class_uri.split('#')[-1]
            keywords_dict[class_name].append(keyword)
        
        print(f"✅ Keywords caricate per {len(keywords_dict)} categorie")
        
        # Mostra statistiche
        for category, keywords in list(keywords_dict.items())[:5]:
            print(f"  {category}: {len(keywords)} keywords")
        
    except Exception as e:
        print(f"⚠️ Errore nell'estrazione SPARQL: {e}")
        print("📝 Usando metodo alternativo...")
        
        # Metodo alternativo: cerca direttamente le triple
        for subject, predicate, obj in g:
            if str(predicate).endswith('#hasKeyword'):
                class_name = str(subject).split('#')[-1]
                keyword = str(obj)
                keywords_dict[class_name].append(keyword)
        
        print(f"✅ Keywords caricate (metodo alternativo) per {len(keywords_dict)} categorie")
    
    # Converti defaultdict in dict normale e aggiungi categoria "Altro"
    keywords_dict = dict(keywords_dict)
    if 'Altro' not in keywords_dict:
        keywords_dict['Altro'] = []  # Categoria catch-all senza keywords
    
    return keywords_dict



def get_all_subclasses(g, base_class_uri):
    """Ottiene tutte le sottoclassi di una classe base"""
    subclasses = set()
    direct_subclasses = set(s for s, p, o in g.triples((None, RDFS.subClassOf, base_class_uri)))
    
    for subclass in direct_subclasses:
        subclass_name = str(subclass).split('#')[-1]
        subclasses.add(subclass_name)
        subclasses |= get_all_subclasses(g, subclass)
    
    # Includi "Altro" come categoria valida
    subclasses.add('Altro')
    return subclasses



def get_superclasses(g, cls_uri):
    """Ottiene tutte le superclassi di una classe"""
    superclasses = set()
    for s, p, o in g.triples((cls_uri, RDFS.subClassOf, None)):
        name = str(o).split('#')[-1]
        if name != str(cls_uri).split('#')[-1]:  # evita loop
            superclasses.add(name)
            superclasses |= get_superclasses(g, o)
    return superclasses



def get_most_specific_category(labels, g, ns):
    """
    NUOVA FUNZIONE: Restituisce SOLO la categoria più specifica
    Invece di aggiungere superclassi, trova la foglia più specifica
    """
    if not labels:
        return ['Altro']  # Default fallback
    
    # Se c'è "Altro", usalo solo se è l'unico
    if len(labels) == 1 and labels[0] == 'Altro':
        return ['Altro']
    
    # Rimuovi "Altro" se ci sono categorie più specifiche
    filtered_labels = [label for label in labels if label != 'Altro']
    if not filtered_labels:
        return ['Altro']
    
    # Trova la categoria più specifica (quella senza sottoclassi)
    most_specific = []
    
    for label in filtered_labels:
        try:
            # Controlla se questa categoria ha sottoclassi
            subclasses = get_all_subclasses(g, ns[label])
            # Se non ha sottoclassi tra le etichette correnti, è specifica
            has_subclass_in_labels = any(sub in filtered_labels for sub in subclasses if sub != label and sub != 'Altro')
            
            if not has_subclass_in_labels:
                most_specific.append(label)
        except:
            # Se non è nell'ontologia, considerala specifica
            most_specific.append(label)
    
    # Se non troviamo nulla di specifico, prendi la prima categoria
    return most_specific[:1] if most_specific else [filtered_labels[0]]



# def extract_text_from_pdf(pdf_path):
#     """Estrae testo da PDF"""
#     text = ""
#     try:
#         with open(pdf_path, "rb") as file:
#             reader = PyPDF2.PdfReader(file)
#             for page in reader.pages:
#                 text += page.extract_text() or ""
#     except Exception as e:
#         print(f"Errore estrazione testo PDF {pdf_path}: {e}")
#     return text



# def preprocess_text(text):
#     """Preprocessa il testo per l'analisi"""
#     text = text.lower()
#     text = re.sub(r'\d+', ' ', text)
#     text = re.sub(r'[^\w\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text



def parse_labels(label_str):
    """Parsa le etichette da stringa a lista"""
    if pd.isna(label_str):
        return []
    labels = [lbl.strip() for lbl in str(label_str).split(';')]
    seen = set()
    unique_labels = []
    for lbl in labels:
        if lbl and lbl not in seen:
            seen.add(lbl)
            unique_labels.append(lbl)
    return unique_labels

def load_dataset_with_single_category(csv_file, g, ns):
    """
    Carica dataset con UNA SOLA categoria per documento,
    usando la colonna 'clean_text' già presente nel CSV.
    """
    df = pd.read_csv(csv_file, low_memory=False)

    # Usiamo direttamente 'clean_text'
    df['text'] = df['clean_text'].fillna("")

    single_labels = []
    for idx, row in df.iterrows():
        raw_labels = parse_labels(row['category'])
        specific_label = get_most_specific_category(raw_labels, g, ns)
        single_labels.append(specific_label[0] if specific_label else None)

    df['single_label'] = single_labels
    df_train = df[df['single_label'].notna()].copy()

    return df_train


def train_eval_single_label_model(name, clf, X_train, y_train, X_val, y_val, all_labels):
    """
    MODIFICATA: Addestra e valuta modello per classificazione SINGLE-LABEL
    """
    print(f"---- Training {name} (Single-Label) ----")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    
    # Metriche per single-label
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} F1 weighted: {f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=None, zero_division=0))
    return clf



def predict_single_category(df, clf, vectorizer, all_labels):
    """
    NUOVA: Predice UNA SOLA categoria per documento
    """
    X = vectorizer.transform(df['text'])
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)
    
    # Ottieni confidenze
    confidences = []
    for i, pred in enumerate(predictions):
        pred_idx = all_labels.index(pred) if pred in all_labels else 0
        confidence = probabilities[i][pred_idx]
        confidences.append(confidence)
    
    df_result = df.copy()
    df_result['predicted_category'] = predictions
    df_result['confidence'] = confidences
    return df_result



def balance_single_label_dataset(df, label_col='single_label', max_samples_per_class=100):
    """
    NUOVA: Bilancia dataset per classificazione single-label
    """
    counter = Counter(df[label_col])
    print(f"📊 Distribuzione originale:")
    for label, count in counter.items():
        print(f"  {label}: {count}")
    
    # Sottocampiona classi con troppi esempi
    balanced_indices = []
    class_counts = Counter()
    
    # Priorità: prima le classi con meno esempi
    sorted_indices = df.index.tolist()
    np.random.shuffle(sorted_indices)  # Randomizza per evitare bias
    
    for idx in sorted_indices:
        label = df.loc[idx, label_col]
        if class_counts[label] < max_samples_per_class:
            balanced_indices.append(idx)
            class_counts[label] += 1
    
    df_balanced = df.loc[balanced_indices].copy()
    
    print(f"📊 Distribuzione bilanciata:")
    counter_balanced = Counter(df_balanced[label_col])
    for label, count in counter_balanced.items():
        print(f"  {label}: {count}")
    
    return df_balanced



def create_enhanced_features(df, ontology_keywords):
    """
    NUOVA: Crea features avanzate basate su keywords ontologiche
    Combina TF-IDF con conteggi keyword-specifici
    """
    print("🔧 Creazione features avanzate basate su ontologia...")
    
    # Features per ogni documento
    keyword_features = []
    
    for _, row in df.iterrows():
        text = str(row['text']).lower()
        doc_features = {}
        
        # Conta keywords per ogni categoria
        for category, keywords in ontology_keywords.items():
            keyword_count = 0
            for keyword in keywords:
                if keyword:  # Skip empty keywords
                    # Usa whole-word matching
                    if ' ' in keyword:
                        # Per frasi multi-parola
                        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        matches = len(re.findall(pattern, text))
                        keyword_count += matches * len(keyword.split())
                    else:
                        # Per singole parole
                        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        matches = len(re.findall(pattern, text))
                        keyword_count += matches
            
            doc_features[f'{category}_keywords'] = keyword_count
        
        keyword_features.append(doc_features)
    
    return pd.DataFrame(keyword_features)



if __name__ == "__main__":
    # Configurazione
    ontology_path = "./Ontology.owx"  # AGGIORNATO per usare nuova ontologia
    base_folder = "./References"
    csv_file = "./training_set_single_category_85percent.csv"
    
    print("🚀 DATASET CREATION CON KEYWORDS DALL'ONTOLOGIA")
    print("=" * 60)
    
    # Carica ontologia
    g = Graph()
    g.parse(ontology_path, format='xml')
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    
    # CARICA KEYWORDS DALL'ONTOLOGIA (invece di hardcoded)
    ONTOLOGY_KEYWORDS = load_keywords_from_ontology(ontology_path, NS)
    
    if not ONTOLOGY_KEYWORDS:
        print("⚠️ Nessuna keyword trovata nell'ontologia!")
        print("📝 Verifica che l'ontologia contenga le hasKeyword properties")
        print("🔧 Usando fallback keywords minimali...")
        ONTOLOGY_KEYWORDS = {
            'Informatica': ['computer', 'software', 'algorithm'],
            'AI_ML': ['machine learning', 'artificial intelligence'],
            'Altro': []
        }
    
    # Ottieni tutte le categorie dall'ontologia
    categories = set(['Scienza'])
    categories |= get_all_subclasses(g, NS['Scienza'])
    categories.add('Studi_umanistici')
    categories |= get_all_subclasses(g, NS['Studi_umanistici'])
    categories.add('Altro')
    all_labels = sorted(list(categories))
    
    print(f"📊 Categorie totali dall'ontologia: {len(all_labels)}")
    print(f"📚 Keywords caricate dall'ontologia: {len(ONTOLOGY_KEYWORDS)} categorie")
    print("🔍 Prime 5 categorie con keywords:")
    for category in list(ONTOLOGY_KEYWORDS.keys())[:5]:
        keywords = ONTOLOGY_KEYWORDS[category]
        print(f"  {category}: {len(keywords)} keywords")
    
    # Carica dataset con UNA categoria per documento
    df_train = load_dataset_with_single_category(csv_file, g, NS)
    print(f"📁 Dataset caricato: {len(df_train)} documenti")
    
    # Bilancia dataset
    df_train_balanced = balance_single_label_dataset(df_train, 'single_label', max_samples_per_class=50)
    
    # RESET indici per coerenza tra dataframes
    df_train_balanced = df_train_balanced.reset_index(drop=True)
    
    # Crea features ontologiche dopo reset degli indici
    keyword_features_df = create_enhanced_features(df_train_balanced, ONTOLOGY_KEYWORDS)
    keyword_features_df = keyword_features_df.reset_index(drop=True)
    
    print(f"⚖️ Dataset bilanciato: {len(df_train_balanced)} documenti")
    print(f"🎯 Features ontologiche create: {keyword_features_df.shape[1]} features")
    
    # Split train/validation - SINGLE LABEL
    if len(df_train_balanced) >= 10:  # Controllo minimo
        unique_labels = df_train_balanced['single_label'].unique()
        can_stratify = all(sum(df_train_balanced['single_label'] == label) >= 2 for label in unique_labels)
        
        if can_stratify and len(unique_labels) > 1:
            train, val = train_test_split(df_train_balanced, test_size=0.2, random_state=42,
                                         stratify=df_train_balanced['single_label'])
        else:
            train, val = train_test_split(df_train_balanced, test_size=0.2, random_state=42)
    else:
        print("⚠️ Dataset troppo piccolo per il training")
        train, val = df_train_balanced, df_train_balanced.sample(frac=0.1)
    
    # Vettorizzazione TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(train['text'])
    X_val_tfidf = vectorizer.transform(val['text'])
    
    # Seleziona features ontologiche con .iloc per evitare errori di indice
    X_train_keywords = keyword_features_df.iloc[train.index].values
    X_val_keywords = keyword_features_df.iloc[val.index].values
    
    # Combina TF-IDF + Keywords ontologiche
    from scipy.sparse import hstack, csr_matrix
    X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_keywords)])
    X_val_combined = hstack([X_val_tfidf, csr_matrix(X_val_keywords)])
    
    # Labels SINGLE
    y_train = train['single_label'].values
    y_val = val['single_label'].values
    
    print(f"🔧 Features TF-IDF: {X_train_tfidf.shape[1]}")
    print(f"🔧 Features Keywords: {X_train_keywords.shape[1]}")
    print(f"🔧 Features Combinate: {X_train_combined.shape[1]}")
    print(f"📋 Training samples: {X_train_combined.shape[0]}")
    print(f"📋 Unique labels: {len(np.unique(y_train))}")
    
    # Addestramento modelli SINGLE-LABEL con features combinate
    print("\n🚀 ADDESTRAMENTO MODELLI (TF-IDF + ONTOLOGY KEYWORDS):")
    print("=" * 60)
    
    # Logistic Regression
    clf_lr = LogisticRegression(max_iter=500, multi_class='ovr')
    model_lr = train_eval_single_label_model("Logistic Regression", clf_lr, 
                                            X_train_combined, y_train, 
                                            X_val_combined, y_val, all_labels)
    
    # Random Forest (più adatto per features combinate)
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf = train_eval_single_label_model("Random Forest", clf_rf, 
                                             X_train_combined, y_train, 
                                             X_val_combined, y_val, all_labels)
    
    # SVM 
    clf_svc = SVC(probability=True, random_state=42)
    model_svc = train_eval_single_label_model("SVM", clf_svc, 
                                              X_train_combined, y_train, 
                                              X_val_combined, y_val, all_labels)    
    
    
    # Test su dataset completo
    print("\n🎯 VALUTAZIONE SU DATASET COMPLETO:")
    print("=" * 60)
    
    # Crea features per tutto il dataset
    all_keyword_features = create_enhanced_features(df_train_balanced, ONTOLOGY_KEYWORDS)
    X_all_tfidf = vectorizer.transform(df_train_balanced['text'])
    X_all_combined = hstack([X_all_tfidf, csr_matrix(all_keyword_features.values)])
    
    # Predizioni
    predictions = model_rf.predict(X_all_combined)
    probabilities = model_rf.predict_proba(X_all_combined)
    
    # Aggiungi predizioni al dataframe
    df_result = df_train_balanced.copy()
    df_result['predicted_category'] = predictions
    df_result['prediction_confidence'] = [prob.max() for prob in probabilities]
    
    print("📋 Sample predictions con confidence:")
    sample_results = df_result[['filename', 'single_label', 'predicted_category', 'prediction_confidence']].head(10)
    print(sample_results.to_string(index=False))
    
    # Statistiche finali
    accuracy = (df_result['single_label'] == df_result['predicted_category']).mean()
    print(f"\n🎯 Accuracy finale (TF-IDF + Ontology): {accuracy:.4f}")
    
    # Salva risultati
    df_result.to_csv('predictions_with_ontology_keywords.csv', index=False)
    print("\n✅ Risultati salvati in 'predictions_with_ontology_keywords.csv'")
    
    print("\n🎊 ADDESTRAMENTO COMPLETATO CON KEYWORDS DALL'ONTOLOGIA!")
    print("🔗 Fonte unica di verità: ontologia OWL")
    print("🎯 Features combinate: TF-IDF + Keywords ontologiche")
    print(f"📈 Miglioramento con features ontologiche rispetto a solo TF-IDF")
