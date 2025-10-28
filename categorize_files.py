import pandas as pd
import re
from collections import defaultdict, Counter
from rdflib import Graph, Namespace, URIRef
import os

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

    # Gerarchia di specificità
    for level in ['very_specific', 'specific']:
        if best_category is None:
            for category in categories_by_specificity[level]:
                score = category_scores.get(category, 0)
                if score > best_score:
                    best_score = score
                    best_category = category

    if best_category is None and extension in extension_categories:
        ext_cats = extension_categories[extension]
        if ext_cats and ext_cats[0] != 'Altro':
            best_category = ext_cats[0]

    if best_category is None:
        for category in categories_by_specificity['general']:
            if category_scores.get(category, 0) > 3:
                best_category = category
                break
    
    if best_category is None:
        best_category = "Altro"
        
    return best_category

def print_statistics(df):
    """Stampa le statistiche di categorizzazione per il DataFrame fornito."""
    category_counts = Counter(df['category'])

    print(f"\n📊 STATISTICHE SUL SET DI DATI CATEGORIZZATO:")
    print("=" * 70)
    print(f"File totali categorizzati: {len(df):,}")
    print(f"\n📈 DISTRIBUZIONE PER CATEGORIA:")
    print("-" * 70)

    counts_by_level = defaultdict(int)
    level_map = {cat: level for level, cats in categories_by_specificity.items() for cat in cats}

    for category, count in category_counts.most_common():
        level_name = level_map.get(category, "fallback").upper()
        counts_by_level[level_name] += count
        percentage = (count / len(df)) * 100
        print(f"{level_name:20} {category:25}: {count:4d} file ({percentage:5.1f}%)")
    
    print("\n🎯 RIEPILOGO PER SPECIFICITÀ:")
    print("-" * 50)
    total_files = len(df)
    for level, count in counts_by_level.items():
        percentage = (count / total_files) * 100
        print(f"{level:20}: {count:4d} ({percentage:5.1f}%)")
    
    specific_total = counts_by_level.get('VERY_SPECIFIC', 0) + counts_by_level.get('SPECIFIC', 0)
    quality_score = (specific_total / total_files) * 100 if total_files > 0 else 0
    print(f"\n📈 QUALITÀ CLASSIFICAZIONE: {quality_score:.1f}% di categorie specifiche")

def categorize_entire_file(input_csv, output_folder, output_file):
    """
    Carica un CSV, categorizza il 100% dei dati e salva il risultato 
    nel file specificato in output_folder/output_file.
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"❌ ERRORE: File di input '{input_csv}' non trovato.")
        return None

    print(f"📁 Trovati {len(df):,} file da categorizzare.")

    # --- CATEGORIZZAZIONE DEL 100% DEL FILE ---
    print("\n🔄 Categorizzazione in corso...")
    df['category'] = df.apply(find_most_specific_category, axis=1)
    print("✅ Categorizzazione completata.")

    # --- SALVATAGGIO NELLA CARTELLA E FILE SPECIFICATI ---
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n📄 File salvato in: {output_path}")

    # --- STATISTICHE SUL RISULTATO ---
    print_statistics(df)

    return df
if __name__ == "__main__":
    # --- DEFINISCI QUI I TASK DA ESEGUIRE ---
    # Ogni elemento è una tupla: (file_input_csv, cartella_output, nome_file_output)
    tasks_to_process = [
        (
            "training_result/output_with_text.csv", 
            "training_result", 
            "training_set_categorized.csv"
        ),
        (
            "test_result/test_data_with_text.csv", 
            "test_result", 
            "test_set_categorized.csv"
        ),
        # Esempio per il terzo task definito in create_csv.py
        (
            "test_result_2/test_data_with_text_2.csv", 
            "test_result_2", 
            "test_set_categorized_2.csv"
        )
        # Puoi aggiungere altri task qui
    ]
    # --- FINE DEFINIZIONE TASK ---

    print("🚀 Avvio categorizzazione (euristica) dei file...")
    total_files_categorized = 0

    # Itera sui task definiti e processali uno per uno
    for input_csv, output_folder, output_file in tasks_to_process:
        print("=" * 80)
        print(f"🚀 CATEGORIZZAZIONE TASK: '{input_csv}'")
        
        df_categorized = categorize_entire_file(input_csv, output_folder, output_file)
        
        if df_categorized is not None:
            total_files_categorized += len(df_categorized)
        
        print("=" * 80)

    print("\n🎉 Tutti i processi di categorizzazione sono stati completati.")
    print("\n" + "--- RIEPILOGO COMPLESSIVO ---".center(50))
    print(f"Numero totale di file categorizzati: {total_files_categorized}")
    print("-" * 50)
