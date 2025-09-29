
import pandas as pd
import re
import random
from collections import defaultdict, Counter

def categorize_all_files_partial_with_altro(percentage=85):
    """
    Categorizza una percentuale di TUTTI i file (non solo PDF)
    Include categorizzazione basata su estensione per file senza titoli chiari
    AGGIORNA: Include la categoria "Altro" per gestire casi edge

    Args:
        percentage (int): Percentuale di file da categorizzare (80-90)
    """

    if not 80 <= percentage <= 90:
        print("ERRORE: La percentuale deve essere tra 80 e 90")
        return

    # Leggi il CSV file
    df = pd.read_csv('training_set.csv')
    print(f"📁 Dataset totale: {len(df):,} file")

    # Categorie richieste - AGGIORNATE con "Altro"
    categories = [
        'Biologia', 'Ambiente', 'Ecologia', 'Chimica',
        'Fisica', 'Energia', 'Spazio', 'Informatica', 'AI_ML', 'Web_development', 'System_programming',
        'Comunicazione', 'Data_analysis', 'Database','Security',
        'Medicina', 'Alimentazione', 'Cardiologia', 'Oncologia', 'Antropologia', 'Archeologia', 'Linguistica', 'Culturale',
        'Filosofia', 'Paleontologia', 'Animale','Botanica', 'Umana', 'Storia', 'antica', 'moderna', 'contemporanea', 'Preistoria', 'Altro'
    ]

    # Sistema di categorizzazione ESTESO - AGGIORNATO con "Altro"
    category_keywords = {
      
        'Biologia': ['biology', 'biological', 'organism', 'cell', 'genetic', 'dna', 'rna', 'protein', 'enzyme', 'evolution', 'molecular', 'bio'],
        'Ambiente': ['environment', 'environmental', 'climate', 'pollution', 'sustainability', 'green', 'ecosystem', 'conservation', 'eco'],
        'Ecologia': ['ecology', 'ecological', 'ecosystem', 'biodiversity', 'habitat', 'species', 'wildlife', 'conservation'],
        'Chimica': ['chemistry', 'chemical', 'compound', 'molecule', 'reaction', 'synthesis', 'catalyst', 'polymer', 'organic'],
        'Fisica': ['physics', 'physical', 'quantum', 'mechanics', 'thermodynamics', 'electromagnetic', 'optics', 'particle'],
        'Energia': ['energy', 'power', 'renewable', 'solar', 'wind', 'nuclear', 'battery', 'fuel', 'electricity'],
        'Spazio': ['space', 'satellite', 'orbit', 'planetary', 'astronomy', 'astrophysics', 'cosmic', 'rocket', 'aerospace'],
        'Informatica': ['computer', 'computing', 'algorithm', 'programming', 'software', 'hardware', 'technology', 'digital', 'it'],
        'AI_ML': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'ai', 'ml', 'classification', 'prediction'],
        'Web_development': ['web', 'website', 'html', 'css', 'javascript', 'frontend', 'backend', 'server', 'browser', 'http'],
        'System_programming': ['system', 'operating system', 'kernel', 'linux', 'unix', 'driver', 'embedded', 'real-time'],
        'Comunicazione': ['communication', 'media', 'social', 'network', 'information', 'signal', 'broadcast', 'telecom'],
        'Data_analysis': ['data', 'analysis', 'statistics', 'analytics', 'visualization', 'mining', 'big data', 'dataset'],
        'Database': ['database', 'sql', 'nosql', 'storage', 'dbms', 'query', 'indexing', 'data management'],
        'Security': ['security', 'cybersecurity', 'encryption', 'authentication', 'firewall', 'forensic', 'cryptography'],
        'Medicina': ['medicine', 'medical', 'health', 'healthcare', 'clinical', 'patient', 'treatment', 'diagnosis'],
        'Alimentazione': ['nutrition', 'food', 'diet', 'meal', 'dietary', 'eating', 'vitamin', 'dietitian'],
        'Cardiologia': ['cardiology', 'heart', 'cardiac', 'cardiovascular', 'coronary', 'artery'],
        'Oncologia': ['oncology', 'cancer', 'tumor', 'malignant', 'chemotherapy', 'radiation'],
        'Antropologia': ['anthropology', 'anthropological', 'human', 'culture', 'society', 'social'],
        'Archeologia': ['archaeology', 'archaeological', 'artifact', 'excavation', 'ancient'],
        'Linguistica': ['linguistic', 'language', 'linguistics', 'sociolinguistics'],
        'Culturale': ['cultural', 'culture', 'ethnography', 'ritual', 'tradition'],
        'Filosofia': ['philosophy', 'philosophical', 'ethics', 'metaphysics', 'logic'],
        'Paleontologia': ['paleontology', 'fossil', 'prehistoric', 'evolution', 'extinct'],
        'Animale': ['animal', 'fossil animal', 'vertebrate', 'mammal'],
        'Botanica': ['plant', 'fossil plant', 'botanical', 'flora'],
        'Umana': ['human evolution', 'hominid', 'ancestor', 'primitive human'],
        'Storia': ['history', 'historical', 'past', 'chronology', 'period', 'era'],
        'antica': ['ancient', 'antiquity', 'classical', 'roman', 'greek'],
        'moderna': ['modern', 'renaissance', 'enlightenment', 'industrial revolution'],
        'contemporanea': ['contemporary', 'modern', '19th', '20th', '21st', 'world war'],
        'Preistoria': ['prehistory', 'prehistoric', 'stone age', 'bronze age', 'iron age'],

        # NUOVA CATEGORIA "Altro"
        'Altro': []
    }

    # Mappatura estensioni -> categorie AGGIORNATA con "Altro"
    extension_categories = {
        # Documenti e testi
        '.pdf': ['Scienza', 'Studi_umanistici'],
        '.doc': ['Scienza', 'Studi_umanistici'],
        '.docx': ['Comunicazione', 'Informatica'],
        '.txt': ['Informatica', 'Comunicazione'],
        '.rtf': ['Comunicazione', 'Informatica'],

        # Web development
        '.html': ['Web_development'],
        '.css': ['Web_development'],
        '.js': ['Web_development'],
        '.php': ['Web_development'],
        '.asp': ['Web_development'],

        # Programmazione
        '.py': ['Informatica', 'AI_ML', 'Data_analysis'],
        '.java': ['Informatica', 'System_programming', 'Mobile_development'],
        '.cpp': ['Informatica', 'System_programming'],
        '.c': ['System_programming', 'Informatica', 'Security'],
        '.cs': ['Informatica', 'System_programming'],
        '.rb': ['Web_development', 'Informatica', 'System_programming'],
        '.go': ['System_programming', 'Web_development', 'Informatica'],
        '.rs': ['System_programming', 'Security', 'Informatica'],

        # Database,
        '.sql': ['Database', 'Informatica'],
        '.db': ['Database', 'Informatica'],
        '.sqlite': ['Database', 'Mobile_development'],

        # Dati
        '.csv': ['Data_analysis'],
        '.xlsx': ['Data_analysis'],
        '.xml': ['Data_analysis'],
        '.json': ['Data_analysis'],

        # Default per estensioni sconosciute - AGGIORNATO
        '.unknown': ['Altro']
    }

    def comprehensive_categorize_paper_with_altro(row):
        """Categorizzazione completa per tutti i tipi di file - AGGIORNATA con Altro"""

        # Estrai informazioni
        titolo = str(row.get('titolo', '')).lower()
        filename = str(row.get('filename', '')).lower()
        extension = str(row.get('extension', '')).lower()
        abstract = str(row.get('abstract', '')).lower() if 'abstract' in row else ''
        clean_text = str(row.get('clean_text', '')).lower() if 'clean_text' in row else ''

        full_text = f"{titolo} {filename} {abstract} {clean_text}"

        # Calcola punteggi basati su parole chiave
        category_scores = defaultdict(int)
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in full_text:
                    weight = len(keyword.split())
                    frequency = full_text.count(keyword)
                    category_scores[category] += weight * frequency

        # Seleziona categorie con punteggio più alto
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        assigned = [cat for cat, score in sorted_categories if score > 0]

        # Se non ci sono abbastanze categorie dal contenuto, usa l'estensione
        if len(assigned) < 3 and extension in extension_categories:
            ext_cats = extension_categories[extension]
            for ext_cat in ext_cats:
                if ext_cat not in assigned:
                    assigned.append(ext_cat)

        # LOGICA AGGIORNATA: Se ancora non ci sono abbastanza categorie, usa "Altro"
        if len(assigned) < 3:
            # Aggiungi "Altro" come categoria generica
            if 'Altro' not in assigned:
                assigned.append('Altro')

            # Aggiungi categorie generiche basate su tipo di file
            if extension.startswith('.'):
                if extension in ['.exe', '.dll', '.sys', '.ko']:
                    defaults = ['System_programming']
                elif extension in ['.log', '.tmp', '.bak']:
                    defaults = ['System_programming']
                elif extension in ['.conf', '.cfg', '.ini']:
                    defaults = ['System_programming']
                else:
                    defaults = ['Altro']
            else:
                defaults = ['Altro']

            for default in defaults:
                if len(assigned) >= 3:
                    break
                if default not in assigned:
                    assigned.append(default)

        # Aggiungi connessioni interdisciplinari - AGGIORNATE
        interdisciplinary = {
            'AI_ML': ['Informatica', 'Data_analysis'],
            'Medicina': ['Biologia', 'Chimica'],
            'Web_development': ['Informatica', 'Comunicazione'],
            'Mobile_development': ['Informatica', 'Comunicazione'],
            'Security': ['Informatica', 'System_programming'],
            'Game_development': ['Informatica', 'Comunicazione'],
            'Database': ['Informatica', 'Data_analysis'],
            'Altro': ['Informatica', 'Comunicazione']  # NUOVO
        }

        for category in assigned[:]:
            if category in interdisciplinary:
                for related in interdisciplinary[category]:
                    if related not in assigned and len(assigned) < 5:
                        assigned.append(related)

        # Assicurati di avere almeno 3 categorie, inclusa "Altro" se necessario
        while len(assigned) < 3:
            remaining = [cat for cat in categories if cat not in assigned]
            if remaining:
                # Prioritizza "Altro" per documenti difficili da classificare
                if 'Altro' in remaining and 'Altro' not in assigned:
                    assigned.append('Altro')
                else:
                    assigned.append(remaining[0])
            else:
                break

        return '; '.join(assigned[:5])

    # Resto del codice rimane uguale...
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

    # Categorizza solo i file selezionati
    print("\n🔄 Categorizzazione in corso...")
    for idx in selected_indices:
        df.at[idx, 'category'] = comprehensive_categorize_paper_with_altro(df.loc[idx])

    # Salva risultato
    output_file = f'training_set_all_files_with_altro_{percentage}percent.csv'
    df.to_csv(output_file, index=False)

    # Statistiche dettagliate
    categorized_df = df[df['category'] != '']
    all_categories = []
    for cat_str in categorized_df['category']:
        if pd.notna(cat_str) and cat_str:
            all_categories.extend(cat_str.split('; '))

    category_counts = Counter(all_categories)

    print(f"\n✅ COMPLETATO! File salvato: {output_file}")
    print(f"\n📊 STATISTICHE FINALI:")
    print("=" * 70)
    print(f"File totali nel dataset: {len(df):,}")
    print(f"File categorizzati: {len(categorized_df):,} ({len(categorized_df)/len(df)*100:.1f}%)")
    print(f"File non categorizzati: {len(df) - len(categorized_df):,}")
    print(f"Media categorie per file: {len(all_categories) / len(categorized_df):.1f}")

    print(f"\n📈 DISTRIBUZIONE PER CATEGORIA (incluso 'Altro'):")
    print("-" * 70)
    in_target = 0
    total_used = 0
    for category in sorted(categories):
        count = category_counts.get(category, 0)
        if count > 0:
            total_used += 1
            if 15 <= count <= 20:
                status = "✅ TARGET"
                in_target += 1
            elif count == 0:
                status = "❌ NON USATA"
            elif count < 15:
                status = f"⬇️ SOTTO ({15-count} mancanti)"
            else:
                status = f"⬆️ SOPRA (+{count-20})"
            print(f"{status:20} {category:25}: {count:3d} file")

    # Statistiche speciali per "Altro"
    altro_count = category_counts.get('Altro', 0)
    print(f"\n🎯 CATEGORIA 'ALTRO': {altro_count} file ({altro_count/len(categorized_df)*100:.1f}% del totale categorizzato)")

    print(f"\n🎯 RIEPILOGO:")
    print("-" * 50)
    print(f"Categorie utilizzate: {total_used}/{len(categories)}")
    print(f"Categorie nel target (15-20): {in_target}")
    print(f"File con minimo 3 categorie: {sum(1 for cat_str in categorized_df['category'] if len(cat_str.split('; ')) >= 3)}")

    return df

# Esegui la categorizzazione completa con "Altro"
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

    print(f"🚀 CATEGORIZZAZIONE COMPLETA AL {percentage}% - CON CATEGORIA 'ALTRO'")
    print("Include TUTTI i tipi di file (PDF, HTML, immagini, codice, etc.)")
    print("NOVITÀ: Categoria 'Altro' per gestire documenti difficili da classificare")
    print("=" * 80)

    # Set seed per riproducibilità
    random.seed(42)
    result = categorize_all_files_partial_with_altro(percentage)
