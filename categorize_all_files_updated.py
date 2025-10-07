import pandas as pd
import re
import random
from collections import defaultdict, Counter
from rdflib import Graph, Namespace, URIRef # --- Estrazione dinamica categoria→keywords dall'ontologia ---


def estrai_category_keywords_da_ontologia(ontology_path):
    g = Graph()
    g.parse(ontology_path, format="xml")  # OWL è RDF/XML

    # Namespace dell'ontologia
    ns = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    hasKeyword = ns.hasKeyword

    category_keywords = defaultdict(list)
    for s, p, o in g.triples((None, hasKeyword, None)):
        cat_name = str(s).split("#")[-1]
        category_keywords[cat_name].append(str(o))
    return dict(category_keywords)

category_keywords = estrai_category_keywords_da_ontologia("Ontology.owx")

# --- Definizione delle categorie per livello di specificità (puoi aggiornarle se cambiano nell'ontologia) ---
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

all_categories = (
    categories_by_specificity['very_specific'] +
    categories_by_specificity['specific'] +
    categories_by_specificity['general'] +
    categories_by_specificity['fallback']
)

# --- Mappatura estensioni -> categoria, lasciata invariata ---
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
    for category in categories_by_specificity['very_specific']:
        score = category_scores.get(category, 0)
        if score > best_score:
            best_score = score
            best_category = category
    if best_category is None:
        for category in categories_by_specificity['specific']:
            score = category_scores.get(category, 0)
            if score > best_score:
                best_score = score
                best_category = category
    if best_category is None and extension in extension_categories:
        ext_categories = extension_categories[extension]
        if ext_categories and ext_categories[0] != 'Altro':
            best_category = ext_categories[0]
    if best_category is None:
        for category in categories_by_specificity['general']:
            score = category_scores.get(category, 0)
            if score > 3:
                best_category = category
                break
    if best_category is None:
        best_category = "Altro"
    return best_category

def categorize_all_files_single_category(percentage=85):
    if not 80 <= percentage <= 90:
        print("ERRORE: La percentuale deve essere tra 80 e 90")
        return

    df = pd.read_csv('output_with_text.csv')
    print(f"📁 Dataset totale: {len(df):,} file")
    files_to_categorize = int(len(df) * percentage / 100)
    print(f"🎯 File da categorizzare: {files_to_categorize:,} ({percentage}% del totale)")

    valid_files = df[
        (df['titolo'].notna()) &
        (df['titolo'].str.strip() != '') &
        (df['titolo'] != ' ')
    ].copy()
    print(f"\n✅ File validi con titolo: {len(valid_files):,}")

    if len(valid_files) >= files_to_categorize:
        selected_indices = random.sample(list(valid_files.index), files_to_categorize)
    else:
        selected_indices = list(valid_files.index)
    print(f"⚠️ Disponibili solo {len(valid_files):,} file validi")
    print(f"🎲 File selezionati: {len(selected_indices):,}")

    df['category'] = ''
    print("\n🔄 Categorizzazione SINGLE-CATEGORY in corso...")
    for idx in selected_indices:
        single_category = find_most_specific_category(df.loc[idx])
        df.at[idx, 'category'] = single_category

    output_file = f'training_set_single_category_{percentage}percent.csv'
    df.to_csv(output_file, index=False)
    categorized_df = df[df['category'] != '']
    category_counts = Counter(categorized_df['category'])

    print(f"\n✅ COMPLETATO! File salvato: {output_file}")
    print(f"\n📊 STATISTICHE FINALI (SINGLE-CATEGORY):")
    print("=" * 70)
    print(f"File totali nel dataset: {len(df):,}")
    print(f"File categorizzati: {len(categorized_df):,} ({len(categorized_df)/len(df)*100:.1f}%)")
    print(f"File non categorizzati: {len(df) - len(categorized_df):,}")
    print(f"Categoria per file: 1 (single-label)")
    print(f"\n📈 DISTRIBUZIONE PER CATEGORIA:")
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
        percentage_of_total = (count / len(categorized_df)) * 100
        print(f"{level} {category:25}: {count:4d} file ({percentage_of_total:5.1f}%)")

    print(f"\n🎯 RIEPILOGO PER SPECIFICITÀ:")
    print("-" * 50)
    print(f"Molto specifiche: {very_specific_count:4d} ({very_specific_count/len(categorized_df)*100:5.1f}%)")
    print(f"Specifiche: {specific_count:4d} ({specific_count/len(categorized_df)*100:5.1f}%)")
    print(f"Generali: {general_count:4d} ({general_count/len(categorized_df)*100:5.1f}%)")
    print(f"Altro: {altro_count:4d} ({altro_count/len(categorized_df)*100:5.1f}%)")
    specific_total = very_specific_count + specific_count
    quality_score = (specific_total / len(categorized_df)) * 100
    print(f"\n📈 QUALITÀ CLASSIFICAZIONE: {quality_score:.1f}% categorie specifiche")
    if quality_score > 80:
        print("✅ OTTIMA qualità - Prevalenza di categorie specifiche")
    elif quality_score > 60:
        print("⚠️ BUONA qualità - Buon bilanciamento")
    else:
        print("❌ MIGLIORABILE - Troppe categorie generali")

    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            percentage = int(sys.argv[1])
        except ValueError:
            percentage = 85
    else:
        percentage = 85
    print(f"🚀 CATEGORIZZAZIONE SINGLE-CATEGORY AL {percentage}%")
    print("NOVITÀ: UNA sola categoria (la più specifica) per ogni file")
    print("Priorità: Categorie foglia > Categorie specifiche > Categorie generali > Altro")
    print("=" * 80)
    random.seed(42)
    result = categorize_all_files_single_category(percentage)
    if result is not None:
        print("\n🎉 Categorizzazione completata con successo!")
    else:
        print("\n❌ Categorizzazione non completata.")