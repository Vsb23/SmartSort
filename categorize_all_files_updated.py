import pandas as pd
import re
import random
from collections import defaultdict, Counter


def categorize_all_files_single_category(percentage=85):
    """
    Categorizza una percentuale di TUTTI i file (non solo PDF)
    MODIFICATO: Assegna SOLO UNA categoria (la più specifica) per ogni file
    
    Args:
        percentage (int): Percentuale di file da categorizzare (80-90)
    """
    
    if not 80 <= percentage <= 90:
        print("ERRORE: La percentuale deve essere tra 80 e 90")
        return

    # Leggi il CSV file
    df = pd.read_csv('output_with_text.csv')
    print(f"📁 Dataset totale: {len(df):,} file")

    # Categorie richieste - ORGANIZZATE per specificità (dalla più specifica alla più generale)
    categories_by_specificity = {
        # Categorie MOLTO SPECIFICHE (foglie dell'albero)
        'very_specific': [
            'AI_ML', 'Web_development', 'System_programming', 'Data_analysis', 'Database', 'Security',
            'Ambiente', 'Ecologia', 'Energia', 'Spazio', 'Alimentazione', 'Cardiologia', 'Oncologia',
            'Archeologia', 'antica', 'moderna', 'contemporanea', 'Preistoria',  'Comunicazione', 'Animale',
            'Botanica','Culturale','Umana'
        ],
        # Categorie SPECIFICHE (nodi intermedi)
        'specific': [
            'Informatica', 'Biologia', 'Fisica', 'Medicina', 'Chimica', 'Antropologia', 'Filosofia', 'Paleontologia', 'Storia'
        ],
        # Categorie GENERALI (rami principali - da evitare quando possibile)
        'general': [
            'Scienza', 'Studi_umanistici'
        ],
        # Categoria FALLBACK
        'fallback': ['Altro']
    }
    
    # Lista piatta di tutte le categorie per backward compatibility
    all_categories = (categories_by_specificity['very_specific'] + 
                     categories_by_specificity['specific'] + 
                     categories_by_specificity['general'] + 
                     categories_by_specificity['fallback'])

    # Sistema di categorizzazione con focus sulla specificità
    category_keywords = {
    'Biologia': ['biology', 'biological', 'organism', 'cellular', 'genetic', 'genome', 'ribonucleic', 'protein', 'enzyme', 'biotechnology'],
    
    'Ambiente': ['environment', 'environmental', 'climate', 'pollution', 'sustainability','green', 'ecosystem', 'conservation', 'biodiversity', 'renewable'],
    
    'Ecologia': ['ecology', 'ecological', 'habitat', 'species', 'wildlife', 'biome', 'population', 'community', 'niche', 'predator'],
    
    'Chimica': ['chemistry', 'chemical', 'compound', 'molecule', 'reaction', 'synthesis', 'catalyst', 'polymer', 'organic', 'inorganic'],
    
    'Fisica': ['physics', 'physical', 'quantum', 'mechanics', 'thermodynamics', 'electromagnetic', 'optics', 'particle', 'relativity', 'gravity'],
    
    'Energia': ['energy', 'power', 'solar', 'wind', 'nuclear', 'battery', 'fuel', 'electricity', 'turbine', 'generator'],
    
    'Spazio': ['space', 'satellite', 'orbit', 'planetary', 'astronomy', 'astrophysics', 'cosmic', 'rocket', 'aerospace', 'telescope'],
    
    'Informatica': ['computer', 'computing', 'algorithm', 'programming', 'software','hardware', 'digital', 'coding', 'processor', 'binary'],
    
    'AI_ML': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'classification', 'prediction', 'supervised', 'unsupervised', 'regression', 'clustering'],
    
    'Web_development': ['website', 'html', 'stylesheet', 'javascript', 'frontend', 'backend', 'server', 'browser', 'http', 'responsive'],
    
    'System_programming': ['system', 'operating system', 'kernel', 'linux', 'unix', 'driver', 'embedded', 'real-time', 'firmware', 'compiler'],
    
    'Comunicazione': ['communication', 'media', 'broadcast', 'information', 'messaging', 'signal', 'telecom', 'wireless', 'protocol', 'transmission'],
    
    'Data_analysis': ['analysis', 'statistics', 'analytics', 'visualization', 'mining', 'dataset', 'metrics', 'correlation', 'trend'],
    
    'Database': ['database', 'query', 'nosql', 'storage', 'dbms', 'indexing', 'table', 'schema', 'transaction', 'relational'],
    
    'Security': ['security', 'cybersecurity', 'encryption', 'authentication', 'firewall','forensic', 'cryptography', 'vulnerability', 'malware', 'intrusion'],
    
    'Medicina': ['medicine', 'medical', 'health', 'healthcare', 'clinical', 'patient', 'treatment', 'diagnosis', 'therapy', 'pharmaceutical'],
    
    'Alimentazione': ['nutrition', 'nutritional', 'food', 'diet', 'meal', 'dietary','eating', 'vitamin', 'dietitian', 'calorie', 'nutrient'],
    
    'Cardiologia': ['cardiology', 'heart', 'cardiac', 'cardiovascular', 'coronary', 'artery', 'blood pressure', 'valve', 'rhythm', 'circulation'],
    
    'Oncologia': ['oncology', 'cancer', 'tumor', 'malignant', 'chemotherapy','radiation', 'metastasis', 'biopsy', 'carcinogen', 'remission'],
    
    'Antropologia': ['anthropology', 'anthropological', 'human', 'society', 'kinship','tribal', 'primitive', 'fieldwork', 'ethnology', 'cultural anthropology'],
    
    'Archeologia': ['archaeology', 'archaeological', 'artifact', 'excavation', 'site','pottery', 'burial', 'stratigraphy', 'dating', 'ruins'],
    
    'Linguistica': ['linguistic', 'language', 'linguistics', 'sociolinguistics', 'phonetics','grammar', 'syntax', 'semantics', 'morphology', 'dialect'],
    
    'Culturale': [
        'cultural', 'folklore', 'tradition', 'custom', 'belief', 
        'identity', 'heritage', 'ceremonial', 'symbolic', 'intercultural'
    ],
    
    'Filosofia': [
        'philosophy', 'philosophical', 'ethics', 'metaphysics', 'logic', 
        'epistemology', 'ontology', 'moral', 'reason', 'wisdom'
    ],
    
    'Paleontologia': [
        'paleontology', 'fossil', 'evolution', 'extinct', 'dinosaur', 
        'paleozoic', 'mesozoic', 'cenozoic', 'sedimentary', 'trilobite'
    ],
    
    'Animale': [
        'animal', 'vertebrate', 'mammal', 'reptile', 'amphibian', 
        'bird', 'fish', 'skeleton', 'bone', 'spine'
    ],
    
    'Botanica': [
        'plant', 'botanical', 'flora', 'leaf', 'root', 
        'stem', 'flower', 'seed', 'photosynthesis', 'chlorophyll'
    ],
    
    'Umana': [
        'human evolution', 'hominid', 'ancestor', 'primitive human', 'homo sapiens', 
        'neanderthal', 'bipedal', 'cranium', 'primates', 'australopithecus'
    ],
    
    'Storia': [
        'history', 'historical', 'past', 'chronology', 'period', 
        'epoch', 'civilization', 'empire', 'dynasty', 'chronicle'
    ],
    
    'antica': [
        'antiquity', 'classical', 'roman', 'greek', 'egypt', 
        'mesopotamia', 'babylon', 'pharaoh', 'gladiator', 'colosseum'
    ],
    
    'moderna': [
        'modern', 'renaissance', 'enlightenment', 'industrial revolution', 'reformation', 
        'capitalism', 'colonialism', 'nationalism', 'democracy', 'monarchy'
    ],
    
    'contemporanea': [
        'contemporary', '19th', '20th', '21st', 'world war', 
        'globalization', 'digitalization', 'internet age', 'terrorism', 'pandemic'
    ],
    
    'Preistoria': [
        'prehistory', 'stone age', 'bronze age', 'iron age', 'neolithic', 
        'paleolithic', 'hunter gatherer', 'cave painting', 'megalith', 'dolmen'
    ],
        'Altro': []  # Categoria catch-all senza keywords specifiche
    }

    # Mappatura estensioni -> categorie SPECIFICHE (non generali)
    extension_categories = {
        '.html': ['Web_development'],
        '.css': ['Web_development'], 
        '.js': ['Web_development'],
        '.php': ['Web_development'],
        '.py': ['AI_ML'],
        '.java': ['System_programming'],
        '.cpp': ['System_programming'],
        '.c': ['System_programming'],
        '.sql': ['Database'],
        '.csv': ['Data_analysis'],
        '.json': ['Data_analysis'],
        '.unknown': ['Altro']
    }

    def find_most_specific_category(row):
        """
        NUOVA FUNZIONE: Trova la categoria PIÙ SPECIFICA per un documento
        Prioritizza le categorie foglia (very_specific) rispetto a quelle generali
        """
        
        # Estrai informazioni
        titolo = str(row.get('titolo', '')).lower()
        filename = str(row.get('filename', '')).lower()
        extension = str(row.get('extension', '')).lower()
        abstract = str(row.get('abstract', '')).lower() if 'abstract' in row else ''
        clean_text = str(row.get('clean_text', '')).lower() if 'clean_text' in row else ''

        full_text = f"{titolo} {filename} {abstract} {clean_text}"
        splitted_clean_text = clean_text.split()
        
        category_scores = defaultdict(int)
        for category, keywords in category_keywords.items():
             if keywords:  # Solo per categorie con keywords definite
        # Nel ciclo delle keywords:
                for keyword in keywords:
                    # Usa regex per parole intere
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    matches = len(re.findall(pattern, full_text.lower()))
                    if matches > 0:
                        weight = len(keyword.split()) * 2
                        category_scores[category] += weight * matches

        
        # STRATEGIA DI SELEZIONE: Priorità alla specificità
        best_category = None
        best_score = 0
        
        # 1. Prima priorità: Categorie MOLTO SPECIFICHE con punteggio > 0
        for category in categories_by_specificity['very_specific']:
            score = category_scores.get(category, 0)
            if score > best_score:
                best_score = score
                best_category = category
        
        # 2. Se non troviamo nulla di molto specifico, cerca nelle SPECIFICHE
        if best_category is None:
            for category in categories_by_specificity['specific']:
                score = category_scores.get(category, 0)
                if score > best_score:
                    best_score = score
                    best_category = category
        
        # 3. Usa estensione solo se non troviamo nulla dalle keywords
        if best_category is None and extension in extension_categories:
            ext_categories = extension_categories[extension]
            if ext_categories and ext_categories[0] != 'Altro':
                best_category = ext_categories[0]
        
        # 4. Come ultima risorsa, usa categoria generale solo se ha un punteggio alto
        if best_category is None:
            for category in categories_by_specificity['general']:
                score = category_scores.get(category, 0)
                if score > 3:  # Soglia più alta per categorie generali
                    best_category = category
                    break
        
        # 5. Fallback finale
        if best_category is None:
            best_category = 'Altro'
        
        return best_category

    # Calcola numero di file da categorizzare
    files_to_categorize = int(len(df) * percentage / 100)
    print(f"🎯 File da categorizzare: {files_to_categorize:,} ({percentage}% del totale)")

    # Seleziona file validi
    valid_files = df[
        (df['titolo'].notna()) &
        (df['titolo'].str.strip() != '') &
        (df['titolo'] != ' ')
    ].copy()

    print(f"\n✅ File validi con titolo: {len(valid_files):,}")

    # Seleziona casualmente
    if len(valid_files) >= files_to_categorize:
        selected_indices = random.sample(list(valid_files.index), files_to_categorize)
    else:
        selected_indices = list(valid_files.index)
        print(f"⚠️ Disponibili solo {len(valid_files):,} file validi")

    print(f"🎲 File selezionati: {len(selected_indices):,}")

    # Inizializza colonna vuota
    df['category'] = ''

    # Categorizza solo i file selezionati - UNA CATEGORIA per file
    print("\n🔄 Categorizzazione SINGLE-CATEGORY in corso...")
    for idx in selected_indices:
        single_category = find_most_specific_category(df.loc[idx])
        df.at[idx, 'category'] = single_category  # UNA sola categoria, no ";"

    # Salva risultato
    output_file = f'training_set_single_category_{percentage}percent.csv'
    df.to_csv(output_file, index=False)

    # Statistiche dettagliate SINGLE-CATEGORY
    categorized_df = df[df['category'] != '']
    
    # Conta categorie (ora è più semplice, una per file)
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
    
    # Statistiche per livello di specificità
    very_specific_count = 0
    specific_count = 0
    general_count = 0
    altro_count = 0
        

    for category, count in category_counts.most_common():

        if category in categories_by_specificity['very_specific']:
            level = "🎯 MOLTO_SPECIFICA"
            very_specific_count += count
        elif category in categories_by_specificity['specific']:
            level = "📊 SPECIFICA     "
            specific_count += count
        elif category in categories_by_specificity['general']:
            level = "⚠️ GENERALE      "
            general_count += count
        else:  # Altro
            level = "🆘 ALTRO         "
            altro_count += count
        
        percentage_of_total = (count / len(categorized_df)) * 100
        print(f"{level} {category:25}: {count:4d} file ({percentage_of_total:5.1f}%)")

    print(f"\n🎯 RIEPILOGO PER SPECIFICITÀ:")
    print("-" * 50)
    print(f"Molto specifiche: {very_specific_count:4d} ({very_specific_count/len(categorized_df)*100:5.1f}%)")
    print(f"Specifiche:       {specific_count:4d} ({specific_count/len(categorized_df)*100:5.1f}%)")
    print(f"Generali:         {general_count:4d} ({general_count/len(categorized_df)*100:5.1f}%)")
    print(f"Altro:            {altro_count:4d} ({altro_count/len(categorized_df)*100:5.1f}%)")
    
    # Verifica qualità: % di categorie specifiche vs generali
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


# Esegui la categorizzazione SINGLE-CATEGORY
if __name__ == "__main__":
    import sys
    
    # Leggi percentuale da argomenti o usa default 85%
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

    # Set seed per riproducibilità
    random.seed(42)
    result = categorize_all_files_single_category(percentage)