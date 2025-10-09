import os
import pandas as pd
import fitz  # PyMuPDF - migliore per estrazione testo
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pickle

class TextExtractor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Carica modello spaCy per processing più avanzato
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("spaCy model non trovato. Installa con: python -m spacy download en_core_web_sm")
            self.nlp = None

    def extract_text_with_pymupdf(self, pdf_path):
        """Estrazione testo con PyMuPDF (più efficace)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Errore nell'estrazione testo con PyMuPDF da {pdf_path}: {str(e)}")
            return ""

    def clean_text(self, text):
        """Pulizia base del testo"""
        if not text:
            return ""
        
        # Rimuovi caratteri speciali e numeri
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Converti in minuscolo
        text = text.lower()
        # Rimuovi spazi multipli
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize_text(self, text):
        """Tokenizzazione del testo"""
        tokens = word_tokenize(text)
        # Rimuovi stopwords e token corti
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        # Lemmatizzazione
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens

    def extract_sections(self, text):
        """Estrazione di sezioni specifiche (titolo, abstract, introduzione)"""
        sections = {
            'title': '',
            'abstract': '',
            'introduction': '',
            'full_text': text[:10000]  # Limita a primi 10k caratteri per efficienza
        }
        
        # Cerca titolo (prima riga significativa)
        lines = text.split('\n')
        for line in lines:
            if len(line.strip()) > 20 and len(line.strip()) < 200:
                sections['title'] = line.strip()
                break
        
        # Cerca abstract
        abstract_patterns = [r'abstract', r'summary', r'abstract—']
        for i, line in enumerate(lines):
            if any(pattern in line.lower() for pattern in abstract_patterns):
                # Prendi le prossime 5-10 righe come abstract
                abstract = ' '.join(lines[i+1:i+10])
                sections['abstract'] = abstract[:500]  # Limita lunghezza
                break
        
        return sections

    def extract_text_features(self, text):
        """Estrazione di features statistiche dal testo"""
        if not text:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'unique_words': 0,
                'lexical_diversity': 0
            }
        
        words = text.split()
        sentences = text.split('.')
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences]) if sentences else 0,
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0
        }
        return features

def process_all_documents(csv_path, output_csv):
    """
    Processa tutti i documenti dal CSV, estraendo testo solo da file unici
    e mappando i risultati su tutte le righe pertinenti.
    """
    df = pd.read_csv(csv_path)
    extractor = TextExtractor()
    
    # Dizionario per memorizzare i dati estratti per ogni percorso di file UNICO
    # Chiave: percorso del file, Valore: dizionario con i dati estratti
    extracted_data_cache = {}
    
    # Liste per popolare le nuove colonne del DataFrame
    full_texts = []
    abstracts = []
    clean_texts = []
    word_counts = []
    unique_words_list = []
    lexical_diversities = []

    for idx, row in df.iterrows():
        # Inizializza i valori di default
        default_data = {'full_text': '', 'abstract': '', 'clean_text': '', 'word_count': 0, 'unique_words': 0, 'lexical_diversity': 0}

        if row['type'] == 'file' and row['extension'] == '.pdf':
            # Costruisci il percorso del file
            relative_path = row['relative_path'] if pd.notna(row['relative_path']) else ''
            relative_path = relative_path.strip()
            if relative_path == '.':
                relative_path = ''
            
            if 'filename' in row and pd.notna(row['filename']) and row['filename'].strip() != '':
                pdf_filename = row['filename']
            else:
                pdf_filename = row['titolo'] + '.pdf'

            pdf_path = os.path.normpath(os.path.join("./References", relative_path, pdf_filename))
            
            # Se il file non è ancora stato processato, estrai i dati e salvali in cache
            if pdf_path not in extracted_data_cache:
                if os.path.exists(pdf_path):
                    text = extractor.extract_text_with_pymupdf(pdf_path)
                    sections = extractor.extract_sections(text)
                    features = extractor.extract_text_features(text)
                    clean_text = extractor.clean_text(text)
                    tokens = extractor.tokenize_text(clean_text)
                    
                    extracted_data_cache[pdf_path] = {
                        'full_text': text,
                        'abstract': sections['abstract'],
                        'clean_text': ' '.join(tokens),
                        'word_count': features['word_count'],
                        'unique_words': features['unique_words'],
                        'lexical_diversity': features['lexical_diversity']
                    }
                else:
                    print(f"File non trovato: {pdf_path}")
                    # Salva in cache il risultato nullo per non cercarlo di nuovo
                    extracted_data_cache[pdf_path] = default_data.copy()
            
            # Recupera i dati dalla cache e li aggiunge alle liste
            data = extracted_data_cache[pdf_path]
            full_texts.append(data['full_text'])
            abstracts.append(data['abstract'])
            clean_texts.append(data['clean_text'])
            word_counts.append(data['word_count'])
            unique_words_list.append(data['unique_words'])
            lexical_diversities.append(data['lexical_diversity'])

        else:
            # Per cartelle o file non PDF, aggiungi valori vuoti
            full_texts.append(default_data['full_text'])
            abstracts.append(default_data['abstract'])
            clean_texts.append(default_data['clean_text'])
            word_counts.append(default_data['word_count'])
            unique_words_list.append(default_data['unique_words'])
            lexical_diversities.append(default_data['lexical_diversity'])
    
    # Aggiungi le nuove colonne al DataFrame originale
    # Questo mantiene l'ordine e il numero di righe originali
    df['full_text'] = full_texts
    df['abstract'] = abstracts
    df['clean_text'] = clean_texts
    df['word_count'] = word_counts
    df['unique_words'] = unique_words_list
    df['lexical_diversity'] = lexical_diversities
    
    # Salva il nuovo CSV
    df.to_csv(output_csv, index=False, escapechar='\\')
    print(f"CSV con testi estratti salvato come: {output_csv}")
    
    return df

def generate_tfidf_matrix(csv_path, output_pickle):
    """Genera matrice TF-IDF per analisi successiva"""
    
    df = pd.read_csv(csv_path)
    
    # Filtra documenti con testo
    text_docs = df[df['clean_text'].notna() & (df['clean_text'] != '')]
    
    if len(text_docs) > 0:
        # Crea vettorizzatore TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(text_docs['clean_text'])
        
        # Salva matrice e vettorizzatore
        with open(output_pickle, 'wb') as f:
            pickle.dump({
                'tfidf_matrix': tfidf_matrix,
                'feature_names': vectorizer.get_feature_names_out(),
                'documents': text_docs['titolo'].tolist()
            }, f)
        
        print(f"Matrice TF-IDF salvata in: {output_pickle}")
        return tfidf_matrix, vectorizer
    else:
        print("Nessun documento con testo trovato.")
        return None, None

if __name__ == "__main__":
    # Processa i documenti
    df = process_all_documents('output.csv', 'output_with_text.csv')
    
    # Genera matrice TF-IDF
    tfidf_matrix, vectorizer = generate_tfidf_matrix('output_with_text.csv', 'tfidf_data.pkl')
    
    print("Estrazione testo e preprocessing completati!")
    if 'clean_text' in df.columns:
        print(f"Documenti processati: {len(df)}")
        print(f"Documenti con testo estratto: {len(df[df['clean_text'] != ''])}")
    else:
        print("La colonna 'clean_text' non è stata creata.")