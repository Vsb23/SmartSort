import os
import pandas as pd
import PyPDF2
import fitz  # PyMuPDF - migliore per estrazione testo
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pdfplumber

# Scarica risorse NLTK (eseguire una volta sola)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

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



    def extract_text_pdfplumber(pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    def extract_text_with_pypdf2(self, pdf_path):
        """Estrazione testo con PyPDF2 (fallback)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Errore nell'estrazione testo con PyPDF2 da {pdf_path}: {str(e)}")
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
            return {}
        
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
    """Processa tutti i documenti dal CSV e aggiunge features di testo"""
    
    # Carica il CSV esistente
    df = pd.read_csv(csv_path)
    extractor = TextExtractor()
    
    text_data = []
    features_list = []
    
    for idx, row in df.iterrows():
        if row['type'] == 'file' and row['extension'] == '.pdf':
            relative_path = row['relative_path']
            if relative_path == '.' or relative_path == '':
                relative_path = ''  # considera la cartella base senza punto
            pdf_path = os.path.join("./References", relative_path, row['titolo'] + '.pdf')
            pdf_path = os.path.normpath(pdf_path)  # pulisce eventuali "/./" o doppi slash

            # Usa file_name se presente e non vuoto, altrimenti titolo + .pdf
            if 'file_name' in row and pd.notna(row['file_name']) and row['file_name'].strip() != '':
                pdf_filename = row['file_name']
            else:
                pdf_filename = row['titolo'] + '.pdf'

            pdf_path = os.path.normpath(os.path.join("./References", relative_path, pdf_filename))
            
            if os.path.exists(pdf_path):
                # Estrai testo
                text = extractor.extract_text_with_pymupdf(pdf_path)
                if not text:
                    text = extractor.extract_text_with_pypdf2(pdf_path)
                if not text:
                    text = extractor.extract_text_pdfplumber(pdf_path)
                
                # Estrai sezioni
                sections = extractor.extract_sections(text)
                
                # Features statistiche
                features = extractor.extract_text_features(text)
                
                # Pulizia e tokenizzazione
                clean_text = extractor.clean_text(text)
                tokens = extractor.tokenize_text(clean_text)
                
                text_data.append({
                    'titolo': row['titolo'],
                    'full_text': text,
                    'abstract': sections['abstract'],
                    'clean_text': ' '.join(tokens),
                    'word_count': features['word_count'],
                    'unique_words': features['unique_words'],
                    'lexical_diversity': features['lexical_diversity']
                })
                
                features_list.append(features)
            else:
                print(f"File non trovato: {pdf_path}")
                text_data.append({
                    'titolo': row['titolo'],
                    'full_text': '',
                    'abstract': '',
                    'clean_text': '',
                    'word_count': 0,
                    'unique_words': 0,
                    'lexical_diversity': 0
                })
        else:
            # Per cartelle o file non PDF
            text_data.append({
                'titolo': row['titolo'],
                'full_text': '',
                'abstract': '',
                'clean_text': '',
                'word_count': 0,
                'unique_words': 0,
                'lexical_diversity': 0
            })
    
    # Crea DataFrame con i testi estratti
    text_df = pd.DataFrame(text_data)
    
    # Unisci con il DataFrame originale
    result_df = pd.merge(df, text_df, on='titolo', how='left')
    
    # Salva il nuovo CSV
    result_df.to_csv(output_csv, index=False)
    print(f"CSV con testi estratti salvato come: {output_csv}")
    
    return result_df

def generate_tfidf_matrix(csv_path, output_pickle):
    """Genera matrice TF-IDF per analisi successiva"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle
    
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
    print(f"Documenti processati: {len(df)}")
    print(f"Documenti con testo estratto: {len(df[df['clean_text'] != ''])}")