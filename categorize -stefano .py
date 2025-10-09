import pandas as pd
import re
import random
from collections import defaultdict, Counter
from rdflib import Graph, Namespace, URIRef
import os # Importato per gestire cartelle e percorsi

def estrai_category_keywords_da_ontologia(ontology_path):
    """Estrae dinamicamente le categorie e le loro keyword da un'ontologia."""
    g = Graph()
    g.parse(ontology_path, format="xml")

    ns = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    hasKeyword = ns.hasKeyword

    category_keywords = defaultdict(list)
    for s, p, o in g.triples((None, hasKeyword, None)):
        cat_name = str(s).split("#")[-1]
        category_keywords[cat_name].append(str(o))
    return dict(category_keywords)

# --- Caricamento dinamico delle keyword dall'ontologia ---
try:
    category_keywords = estrai_category_keywords_da_ontologia("Ontology.owx")
except FileNotFoundError:
    print("❌ ERRORE: File 'Ontology.owx' non trovato. Assicurati che sia nella stessa cartella dello script.")
    exit()


# --- Definizione delle categorie per livello di specificità ---
categories_by_specificity = {
    "very_specific": [
        "AI_ML", "Web_development", "System_programming", "Data_analysis", "Database", "Security",
        "Ambiente", "Ecologia", "Energia", "Spazio", "Alimentazione", "Cardiologia", "Oncologia",
        "Archeologia", "Antica", "Moderna", "Contemporanea", "Preistoria", "Comunicazione", "Animale",
        "Botanica", "Culturale", "Umana"
    ],
    "specific": [
        "Informatica", "Biologia", "Fisica", "Medicina", "Chimica", "Antropologia", "Filosofia", "Paleontologia", "Storia"
    ],
    "general": [
        "Scienza", "Studi_umanistici"
    ],
    "fallback": ["Altro"]
}

# --- Mappatura estensioni -> categoria ---
extension_categories = {
    ".html": ["Web_development"],
    ".css": ["Web_development"],
    ".js": ["Web_development"],
    ".php": ["Web_development"],
    ".py": ["AI_ML"],
    ".java": ["System_programming"],
    ".cpp": ["System_programming"],
    ".c": ["System_programming"],
    ".sql": ["Database"],
    ".csv": ["Data_analysis"],
    ".json": ["Data_analysis"],
    ".unknown": ["Altro"]
}

def find_most_specific_category(row):
    """
    Trova la categoria più specifica per una riga del DataFrame basandosi su
    keyword, testo e metadati, con una gerarchia di specificità.
    """
    titolo = str(row.get('titolo', '')).lower()
    filename = str(row.get('filename', '')).lower()
    extension = str(row.get('extension', '')).lower()
    abstract = str(row.get('abstract', '')).lower() if 'abstract' in row else ''
    clean_text = str(row.get('clean_text', '')).lower() if 'clean_text' in row else ''
    full_text = f"{titolo} {filename} {abstract} {clean_text}"

    category_scores = defaultdict(int)
    for category, keywords in category_keywords.items():
        if keywords:
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = len(re.findall(pattern, full_text.lower()))
                if matches > 0:
                    weight = len(keyword.split()) * 2
                    category_scores[category] += weight * matches

    best_category = None
    best_score = 0

    # 1. Priorità alle categorie molto specifiche
    for category in categories_by_specificity['very_specific']:
        score = category_scores.get(category, 0)
        if score > best_score:
            best_score = score
            best_category = category
    
    # 2. Se nessuna, passa alle categorie specifiche
    if best_category is None:
        for category in categories_by_specificity['specific']:
            score = category_scores.get(category, 0)
            if score > best_score:
                best_score = score
                best_category = category

    # 3. Se nessuna, considera l'estensione del file come fallback
    if best_category is None and extension in extension_categories:
        ext_categories = extension_categories[extension]
        if ext_categories and ext_categories[0] != 'Altro':
            best_category = ext_categories[0]

    # 4. Se ancora nessuna, cerca una categoria generale con un punteggio minimo
    if best_category is None:
        for category in categories_by_specificity['general']:
            score = category_scores.get(category, 0)
            if score > 3: # Soglia minima per evitare assegnazioni casuali
                best_category = category
                break
    
    # 5. Categoria di fallback finale
    if best_category is None:
        best_category = "Altro"
        
    return best_category

def categorize_and_split_files(percentage=85):
    """
    Categorizza una percentuale di file da un CSV e li divide in due set:
    training (categorizzati) e test (non categorizzati), salvandoli in cartelle separate.
    """
    if not 80 <= percentage <= 90:
        print("ERRORE: La percentuale deve essere compresa tra 80 e 90.")
        return

    try:
        df = pd.read_csv('output_with_text.csv')
    except FileNotFoundError:
        print("❌ ERRORE: File 'output_with_text.csv' non trovato. Esegui prima lo script di estrazione del testo.")
        return

    print(f"📁 Dataset totale: {len(df):,} file")
    files_to_categorize = int(len(df) * percentage / 100)
    print(f"🎯 File da usare per il training set: {files_to_categorize:,} ({percentage}% del totale)")

    valid_files = df[df['titolo'].notna() & (df['titolo'].str.strip() != '')].copy()
    print(f"\n✅ File validi con un titolo: {len(valid_files):,}")

    if len(valid_files) < files_to_categorize:
        print(f"⚠️ ATTENZIONE: Disponibili solo {len(valid_files):,} file validi, che sono meno dei {files_to_categorize:,} richiesti.")
        print(f"🎲 Verranno usati tutti i {len(valid_files):,} file validi per il training set.")
        selected_indices = list(valid_files.index)
    else:
        selected_indices = random.sample(list(valid_files.index), files_to_categorize)
    
    print(f"⚙️ File selezionati per la categorizzazione: {len(selected_indices):,}")

    df['category'] = ''
    print("\n🔄 Categorizzazione in corso...")
    for idx in selected_indices:
        single_category = find_most_specific_category(df.loc[idx])
        df.at[idx, 'category'] = single_category

    # --- NUOVA SEZIONE: Divisione e salvataggio in cartelle separate ---
    
    # 1. Creazione delle cartelle di output
    training_folder = "training_data"
    test_folder = "test_data"
    os.makedirs(training_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # 2. Divisione del DataFrame in training e test set
    training_df = df[df['category'] != ''].copy()
    test_df = df[df['category'] == ''].copy()
    
    # Rimuoviamo la colonna 'category' (vuota) dal test set, poiché è superflua
    if 'category' in test_df.columns:
        test_df = test_df.drop(columns=['category'])

    # 3. Definizione dei percorsi di output
    training_output_file = os.path.join(training_folder, f'training_set_{percentage}percent.csv')
    test_output_file = os.path.join(test_folder, f'test_set_{100 - percentage}percent.csv')

    # 4. Salvataggio dei due file CSV
    training_df.to_csv(training_output_file, encoding='utf-8', index=False)
    test_df.to_csv(test_output_file,encoding='utf-8', index=False)

    print(f"\n✅ COMPLETATO!")
    print(f"📄 Training set salvato in: {training_output_file}")
    print(f"📄 Test set salvato in:   {test_output_file}")
    
    # --- STATISTICHE (basate solo sul training set creato) ---
    category_counts = Counter(training_df['category'])

    print(f"\n📊 STATISTICHE SUL TRAINING SET:")
    print("=" * 70)
    print(f"File totali nel training set: {len(training_df):,}")
    print(f"File totali nel test set: {len(test_df):,}")
    print(f"Categoria per file: 1 (single-label)")
    print(f"\n📈 DISTRIBUZIONE PER CATEGORIA (Training Set):")
    print("-" * 70)

    very_specific_count = 0
    specific_count = 0
    general_count = 0
    altro_count = 0
    for category, count in category_counts.most_common():
        if category in categories_by_specificity['very_specific']:
            level = "🎯 MOLTO_SPECIFICA"
            very_specific_count += count
        elif category in categories_by_specificity['specific']:
            level = "📊 SPECIFICA "
            specific_count += count
        elif category in categories_by_specificity['general']:
            level = "⚠️ GENERALE "
            general_count += count
        else:
            level = "🆘 ALTRO "
            altro_count += count
        percentage_of_total = (count / len(training_df)) * 100
        print(f"{level} {category:25}: {count:4d} file ({percentage_of_total:5.1f}%)")

    print(f"\n🎯 RIEPILOGO PER SPECIFICITÀ (Training Set):")
    print("-" * 50)
    print(f"Molto specifiche: {very_specific_count:4d} ({very_specific_count/len(training_df)*100:5.1f}%)")
    print(f"Specifiche: {specific_count:4d} ({specific_count/len(training_df)*100:5.1f}%)")
    print(f"Generali: {general_count:4d} ({general_count/len(training_df)*100:5.1f}%)")
    print(f"Altro: {altro_count:4d} ({altro_count/len(training_df)*100:5.1f}%)")
    specific_total = very_specific_count + specific_count
    quality_score = (specific_total / len(training_df)) * 100
    print(f"\n📈 QUALITÀ CLASSIFICAZIONE: {quality_score:.1f}% di categorie specifiche nel training set")
    if quality_score > 80:
        print("✅ OTTIMA qualità - Prevalenza di categorie specifiche")
    elif quality_score > 60:
        print("⚠️ BUONA qualità - Buon bilanciamento")
    else:
        print("❌ MIGLIORABILE - Troppe categorie generali o 'Altro'")

    return training_df, test_df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            percentage = int(sys.argv[1])
        except ValueError:
            print("Input non valido, uso il valore di default (85%).")
            percentage = 85
    else:
        percentage = 85
        
    print(f"🚀 AVVIO CREAZIONE TRAINING E TEST SET ({percentage}%) 🚀")
    print("=" * 80)
    # random.seed(42) # Per la riproducibilità dei risultati
    
    result = categorize_and_split_files(percentage)
    
    if result is not None:
        print("\n🎉 Operazione completata con successo!")
    else:
        print("\n❌ Operazione non completata a causa di un errore.")