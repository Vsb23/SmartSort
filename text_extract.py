import os
import pandas as pd
import fitz  # PyMuPDF
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class TextExtractor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            print("Modello spaCy non trovato. Esegui: python -m spacy download en_core_web_sm")
            self.nlp = None

    def extract_text_with_pymupdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            print(f"Errore nell'estrazione del testo da {pdf_path}: {e}")
            return ""

    def clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, text):
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return tokens

    def extract_sections(self, text):
        sections = {'title': '', 'abstract': '', 'introduction': ''}
        lines = text.split('\n')
        if lines:
            for line in lines:
                if 20 < len(line.strip()) < 200:
                    sections['title'] = line.strip()
                    break
        abstract_match = re.search(r'abstract[\s\S]{1,1000}', text, re.IGNORECASE)
        if abstract_match:
            sections['abstract'] = abstract_match.group(0)[:500]
        return sections

    def extract_text_features(self, text):
        if not text:
            return {'word_count': 0, 'unique_words': 0, 'lexical_diversity': 0}
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        return {
            'word_count': word_count,
            'unique_words': unique_words,
            'lexical_diversity': unique_words / word_count if word_count > 0 else 0
        }

def process_all_documents(csv_path, output_csv, base_data_folder):
    """
    Processa i documenti da un CSV, estraendo testo dalla cartella base specificata.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ ERRORE: File di input '{csv_path}' non trovato. Salto questo processo.")
        return None

    print(f"INFO: Inizio processo per '{base_data_folder}'. Righe totali: {len(df)}")
    
    # Filtra righe corrotte
    original_rows = len(df)
    is_folder = df['type'] == 'folder'
    is_valid_file = (df['type'] == 'file') & (df['filename'].notna()) & (df['filename'].str.strip() != '')
    df = df[is_folder | is_valid_file].copy()
    if len(df) < original_rows:
        print(f"✅ PULIZIA: Rimosse {original_rows - len(df)} righe corrotte.")

    extractor = TextExtractor()
    extracted_data_cache = {}
    
    new_columns = {
        'full_text': [], 'abstract': [], 'clean_text': [], 'word_count': [],
        'unique_words': [], 'lexical_diversity': []
    }
    
    default_data = {'full_text': '', 'abstract': '', 'clean_text': '', 'word_count': 0, 'unique_words': 0, 'lexical_diversity': 0}

    for _, row in df.iterrows():
        if row['type'] == 'file' and row['extension'] == '.pdf':
            relative_path = row['relative_path'] if pd.notna(row['relative_path']) and row['relative_path'].strip() != '.' else ''
            pdf_filename = row['filename']
            # << MODIFICA >>: Usa il parametro 'base_data_folder' per costruire il percorso
            pdf_path = os.path.normpath(os.path.join(base_data_folder, relative_path, pdf_filename))

            if pdf_path not in extracted_data_cache:
                if os.path.exists(pdf_path):
                    text = extractor.extract_text_with_pymupdf(pdf_path)
                    sections = extractor.extract_sections(text)
                    features = extractor.extract_text_features(text)
                    tokens = extractor.tokenize_text(extractor.clean_text(text))
                    
                    extracted_data_cache[pdf_path] = {
                        'full_text': text, 'abstract': sections['abstract'],
                        'clean_text': ' '.join(tokens), 'word_count': features['word_count'],
                        'unique_words': features['unique_words'], 'lexical_diversity': features['lexical_diversity']
                    }
                else:
                    print(f"File non trovato: {pdf_path}")
                    extracted_data_cache[pdf_path] = default_data.copy()
            
            data = extracted_data_cache[pdf_path]
        else:
            data = default_data.copy()

        for key, value in data.items():
            new_columns[key].append(value)
            
    for col_name, col_values in new_columns.items():
        df[col_name] = col_values
    
    df.to_csv(output_csv, index=False, escapechar='\\')
    print(f"✅ CSV con testi estratti salvato come: {output_csv}")
    return df

def generate_tfidf_matrix(csv_path, output_pickle):
    """Genera matrice TF-IDF solo per i dati di training."""
    df = pd.read_csv(csv_path)
    text_docs = df[df['clean_text'].notna() & (df['clean_text'] != '')]
    
    if not text_docs.empty:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(text_docs['clean_text'])
        with open(output_pickle, 'wb') as f:
            pickle.dump({'tfidf_matrix': tfidf_matrix, 'feature_names': vectorizer.get_feature_names_out()}, f)
        print(f"✅ Matrice TF-IDF per il training set salvata in: {output_pickle}")
    else:
        print("Nessun documento di training con testo trovato per generare la matrice TF-IDF.")

def process_single_task(input_csv, output_csv, data_folder):
    """
    Funzione wrapper per eseguire process_all_documents come un singolo task.
    Restituisce il DataFrame processato o None.
    """
    print(f"\n--- Inizio processo per: '{data_folder}' -> '{output_csv}' ---")
    
    processed_df = process_all_documents(input_csv, output_csv, data_folder)
    
    if processed_df is not None:
        print(f"File processati: {len(processed_df)}")
    else:
        print(f"❌ Task saltato a causa di errore (es. file input non trovato).")
    
    print("-" * 70)
    return processed_df

if __name__ == "__main__":
    # --- DEFINISCI QUI I TASK DA ESEGUIRE ---
    # Ogni tupla: (file_csv_input, file_csv_output, cartella_sorgente_pdf, is_training)
    # is_training=True: genera la matrice TF-IDF per questo set
    tasks_to_process = [
        (
            './training_result/output.csv',
            './training_result/output_with_text.csv',
            './training_data',
            True  # Questo è il set di training
        ),
        (
            './test_result/test_output.csv',
            './test_result/test_data_with_text.csv',
            './test_data',
            False # Questo è il set di test
        ),
        # Esempio per il terzo task definito in create_csv.py
        (
            './test_result_2/test_output_2.csv', 
            './test_result_2/test_data_with_text_2.csv',
            './test_data_2',
            False
        )
        # Puoi aggiungere altri task qui
    ]
    # --- FINE DEFINIZIONE TASK ---
    
    tfidf_pickle_file = 'tfidf_data.pkl'
    # Salva il percorso di output del training set per la generazione TF-IDF
    training_data_path_for_tfidf = None 
    total_files_processed = 0

    print("🚀 Inizio processo di estrazione testo...")
    
    # 1. Itera sui task definiti e processali uno per uno
    for input_csv, output_csv, data_folder, is_training in tasks_to_process:
        
        processed_df = process_single_task(input_csv, output_csv, data_folder)
        
        if processed_df is not None:
            total_files_processed += len(processed_df)
            if is_training:
                # Se questo è il set di training, salva il suo percorso di output
                training_data_path_for_tfidf = output_csv
                print(f"ℹ️  Set di training '{output_csv}' designato per la generazione TF-IDF.")

    print("\n" + "--- FASE FINALE ---".center(70, "="))

    # 2. Genera la matrice TF-IDF SOLO sul set di training (dopo tutti i task)
    if training_data_path_for_tfidf:
        print("Generazione matrice TF-IDF (solo su training)...")
        generate_tfidf_matrix(training_data_path_for_tfidf, tfidf_pickle_file)
    else:
        print("⚠️  Nessun task è stato contrassegnato come 'is_training=True'.")
        print("⚠️  Matrice TF-IDF non generata.")

    print("\n🎉 Estrazione testo e preprocessing completati per tutti i set!")
    print("\n" + "--- RIEPILOGO COMPLESSIVO ---".center(50))
    print(f"Numero totale di record processati: {total_files_processed}")
    print("-" * 50)