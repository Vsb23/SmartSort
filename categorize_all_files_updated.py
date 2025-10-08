import pandas as pd
import re
import random
from collections import defaultdict, Counter
from rdflib import Graph, Namespace, URIRef

def load_keywords_from_ontology(ontology_path):
    print("📖 Caricamento keywords dall'ontologia...")
    g = Graph()
    g.parse(ontology_path, format='xml')
    ns = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    hasKeyword = ns.hasKeyword
    category_keywords = defaultdict(list)
    for s, p, o in g.triples((None, hasKeyword, None)):
        cat_name = str(s).split('#')[-1]
        category_keywords[cat_name].append(str(o))
    print(f"✅ Keywords caricate per {len(category_keywords)} categorie")
    return dict(category_keywords)

category_keywords = load_keywords_from_ontology("Ontology.owx")

# Definizione delle categorie per livello di specificità (aggiorna se cambia ontologia)
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
                matches = len(re.findall(pattern, full_text))
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

def categorize_all_files_single_category_percentage(df, percentage=85):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(len(df) * percentage / 100)
    training_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    training_df['category'] = training_df['category'].astype(object)

    for idx in training_df.index:
        training_df.at[idx, 'category'] = find_most_specific_category(training_df.loc[idx])

    train_file = f"train_set_{percentage}percent.csv"
    test_file = f"test_set_{100 - percentage}percent.csv"

    training_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"File di training salvato: {train_file}, record: {len(training_df)}")
    print(f"File di test salvato: {test_file}, record: {len(test_df)}")

    return train_file, test_file


if __name__ == "__main__":
    csv_path = "./output_with_text.csv"
    df_files = pd.read_csv(csv_path)
    train_csv, test_csv = categorize_all_files_single_category_percentage(df_files, 85)
