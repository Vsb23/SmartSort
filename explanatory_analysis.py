import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle

def exploratory_analysis(csv_path):
    """Analisi esplorativa dei testi estratti"""
    
    df = pd.read_csv(csv_path)
    
    print("=== ANALISI ESPLORATIVA DEI DOCUMENTI ===")
    print(f"Totale documenti: {len(df)}")
    print(f"Documenti con testo: {len(df[df['clean_text'] != ''])}")
    print(f"Documenti senza testo: {len(df[df['clean_text'] == ''])}")
    
    # Statistiche base
    text_docs = df[df['clean_text'] != '']
    if len(text_docs) > 0:
        print("\n=== STATISTICHE TESTI ===")
        print(f"Lunghezza media testo: {text_docs['word_count'].mean():.0f} parole")
        print(f"Vocabolario unico medio: {text_docs['unique_words'].mean():.0f} parole")
        print(f"Diversità lessicale media: {text_docs['lexical_diversity'].mean():.3f}")
        
        # Visualizzazioni
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(text_docs['word_count'], bins=20, alpha=0.7)
        plt.title('Distribuzione lunghezza documenti')
        plt.xlabel('Numero parole')
        plt.ylabel('Frequenza')
        
        plt.subplot(1, 3, 2)
        plt.hist(text_docs['lexical_diversity'], bins=20, alpha=0.7)
        plt.title('Diversità lessicale')
        plt.xlabel('Diversità')
        plt.ylabel('Frequenza')
        
        plt.subplot(1, 3, 3)
        # Word Cloud
        all_text = ' '.join(text_docs['clean_text'].astype(str))
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        
        plt.tight_layout()
        plt.savefig('text_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def load_tfidf_analysis(pickle_path):
    """Analisi della matrice TF-IDF"""
    
    with open(pickle_path, 'rb') as f:
        tfidf_data = pickle.load(f)
    
    tfidf_matrix = tfidf_data['tfidf_matrix']
    feature_names = tfidf_data['feature_names']
    
    print("\n=== ANALISI TF-IDF ===")
    print(f"Dimensioni matrice: {tfidf_matrix.shape}")
    print(f"Numero features: {len(feature_names)}")
    
    # Termini più importanti
    sums = tfidf_matrix.sum(axis=0)
    data = []
    for col, term in enumerate(feature_names):
        data.append((term, sums[0, col]))
    
    ranked_terms = sorted(data, key=lambda x: x[1], reverse=True)[:20]
    
    print("\nTop 20 termini più importanti:")
    for term, score in ranked_terms:
        print(f"{term}: {score:.4f}")

if __name__ == "__main__":
    # Esegui analisi esplorativa
    exploratory_analysis('output_with_text.csv')
    
    # Analisi TF-IDF
    try:
        load_tfidf_analysis('tfidf_data.pkl')
    except:
        print("File TF-IDF non trovato. Esegui prima text_extraction.py")