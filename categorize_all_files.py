
import pandas as pd
import re
import random
from collections import defaultdict, Counter

def categorize_all_files_partial(percentage=25):
    """
    Categorizza una percentuale di TUTTI i file (non solo PDF)
    Include categorizzazione basata su estensione per file senza titoli chiari

    Args:
        percentage (int): Percentuale di file da categorizzare (20-30)
    """

    if not 20 <= percentage <= 30:
        print("ERRORE: La percentuale deve essere tra 20 e 30")
        return

    # Leggi il CSV file
    df = pd.read_csv('training_set.csv')
    print(f"📁 Dataset totale: {len(df):,} file")

    # Categorie richieste
    categories = [
        'Architettura', 'Biologia', 'Ambiente', 'Ecologia', 'Chimica', 'Economia', 'Politiche_di_stato', 
        'Fisica', 'Energia', 'Spazio', 'Informatica', 'AI_ML', 'Web_development', 'System_programming', 
        'Comunicazione', 'Data_analysis', 'Database', 'Game_development', 'Security', 'Mobile_development',
        'Medicina', 'Alimentazione', 'Cardiologia', 'Oncologia', 'Antropologia', 'Archeologia', 
        'Linguistica', 'Culturale', 'Filosofia', 'Etica', 'Politica', 'Metafisica', 'Paleontologia', 'Animale', 
        'Botanica', 'Umana', 'Storia', 'antica', 'moderna', 'contemporanea', 'Preistoria'
    ]

    # Calcola file da categorizzare
    files_to_categorize = int(len(df) * percentage / 100)
    print(f"🎯 File da categorizzare: {files_to_categorize:,} ({percentage}% del totale)")

    # Analisi tipi di file presenti
    print(f"\n📊 ANALISI TIPI DI FILE:")
    print("-" * 40)

    file_types = df['type'].value_counts() if 'type' in df.columns else {'file': len(df)}
    for ftype, count in file_types.items():
        print(f"{ftype:10}: {count:,} file")

    extensions = df['extension'].value_counts() if 'extension' in df.columns else {'.pdf': len(df)}
    print(f"\n📄 ESTENSIONI PIÙ COMUNI:")
    for ext, count in list(extensions.head(10).items()):
        print(f"{ext:8}: {count:,} file")

    # Seleziona file validi (esclude solo cartelle vuote)
    valid_files = df[
        (df['titolo'].notna()) & 
        (df['titolo'].str.strip() != '') & 
        (df['titolo'] != ' ')
    ].copy()

    print(f"\n✅ File validi con titolo: {len(valid_files):,}")

    # Se non ci sono abbastanza file con titolo, include anche quelli senza
    if len(valid_files) < files_to_categorize:
        additional_files = df[
            (~df.index.isin(valid_files.index)) &
            (df['type'] == 'file') if 'type' in df.columns else ~df.index.isin(valid_files.index)
        ]
        additional_needed = min(files_to_categorize - len(valid_files), len(additional_files))
        if additional_needed > 0:
            additional_selected = additional_files.sample(n=additional_needed, random_state=42)
            valid_files = pd.concat([valid_files, additional_selected])
        print(f"➕ File aggiuntivi inclusi: {additional_needed:,}")

    # Seleziona casualmente
    if len(valid_files) >= files_to_categorize:
        selected_indices = random.sample(list(valid_files.index), files_to_categorize)
    else:
        selected_indices = list(valid_files.index)
        print(f"⚠️  Disponibili solo {len(valid_files):,} file validi")

    print(f"🎲 File selezionati: {len(selected_indices):,}")

    # Sistema di categorizzazione ESTESO per tutti i tipi di file
    category_keywords = {
        'Architettura': ['architecture', 'architectural', 'building', 'construction', 'design', 'urban', 'structural', 'cad', 'blueprint'],
        'Biologia': ['biology', 'biological', 'organism', 'cell', 'genetic', 'dna', 'rna', 'protein', 'enzyme', 'evolution', 'molecular', 'bio'],
        'Ambiente': ['environment', 'environmental', 'climate', 'pollution', 'sustainability', 'green', 'ecosystem', 'conservation', 'eco'],
        'Ecologia': ['ecology', 'ecological', 'ecosystem', 'biodiversity', 'habitat', 'species', 'wildlife', 'conservation'],
        'Chimica': ['chemistry', 'chemical', 'compound', 'molecule', 'reaction', 'synthesis', 'catalyst', 'polymer', 'organic'],
        'Economia': ['economy', 'economic', 'market', 'finance', 'business', 'trade', 'investment', 'banking', 'financial'],
        'Politiche_di_stato': ['policy', 'government', 'state', 'public', 'administration', 'governance', 'regulation', 'law'],
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
        'Game_Development': ['game', 'gaming', 'unity', 'graphics', '3d', 'simulation', 'virtual reality', 'interactive'],
        'Security': ['security', 'cybersecurity', 'encryption', 'authentication', 'firewall', 'forensic', 'cryptography'],
        'Mobile_Development': ['mobile', 'android', 'ios', 'smartphone', 'app', 'tablet', 'wireless'],
        'Medicina': ['medicine', 'medical', 'health', 'healthcare', 'clinical', 'patient', 'treatment', 'diagnosis'],
        'Alimentazione': ['nutrition', 'food', 'diet', 'meal', 'dietary', 'eating', 'vitamin', 'dietitian'],
        'Cardiologia': ['cardiology', 'heart', 'cardiac', 'cardiovascular', 'coronary', 'artery'],
        'Oncologia': ['oncology', 'cancer', 'tumor', 'malignant', 'chemotherapy', 'radiation'],
        'Antropologia': ['anthropology', 'anthropological', 'human', 'culture', 'society', 'social'],
        'Archeologia': ['archaeology', 'archaeological', 'artifact', 'excavation', 'ancient'],
        'Linguistica': ['linguistic', 'language', 'linguistics', 'sociolinguistics'],
        'Culturale': ['cultural', 'culture', 'ethnography', 'ritual', 'tradition'],
        'Filosofia': ['philosophy', 'philosophical', 'ethics', 'metaphysics', 'logic'],
        'Etica': ['ethics', 'moral', 'morality', 'virtue', 'justice'],
        'Politica': ['political philosophy', 'democracy', 'liberty', 'freedom'],
        'Metafisica': ['metaphysics', 'ontology', 'being', 'reality', 'existence'],
        'Paleontologia': ['paleontology', 'fossil', 'prehistoric', 'evolution', 'extinct'],
        'Animale': ['animal', 'fossil animal', 'vertebrate', 'mammal'],
        'Botanica': ['plant', 'fossil plant', 'botanical', 'flora'],
        'Umana': ['human evolution', 'hominid', 'ancestor', 'primitive human'],
        'Storia': ['history', 'historical', 'past', 'chronology', 'period', 'era'],
        'antica': ['ancient', 'antiquity', 'classical', 'roman', 'greek'],
        'moderna': ['modern', 'renaissance', 'enlightenment', 'industrial revolution'],
        'contemporanea': ['contemporary', 'modern', '19th', '20th', '21st', 'world war'],
        'Preistoria': ['prehistory', 'prehistoric', 'stone age', 'bronze age', 'iron age']
    }

    # Mappatura estensioni -> categorie per file senza titoli chiari
    extension_categories = {
        # Documenti e testi
        '.pdf': ['Informatica', 'Data_analysis', 'Comunicazione'],
        '.doc': ['Comunicazione', 'Informatica', 'Data_analysis'],
        '.docx': ['Comunicazione', 'Informatica', 'Data_analysis'],
        '.txt': ['Informatica', 'Comunicazione', 'Data_analysis'],
        '.rtf': ['Comunicazione', 'Informatica', 'Data_analysis'],

        # Web development
        '.html': ['Web_development', 'Informatica', 'Comunicazione'],
        '.css': ['Web_development', 'Informatica', 'System_programming'],
        '.js': ['Web_development', 'Informatica', 'AI_ML'],
        '.php': ['Web_development', 'Database', 'Informatica'],
        '.asp': ['Web_development', 'Database', 'Informatica'],

        # Programmazione
        '.py': ['Informatica', 'AI_ML', 'Data_analysis'],
        '.java': ['Informatica', 'System_programming', 'Mobile_Development'],
        '.cpp': ['Informatica', 'System_programming', 'Game_Development'],
        '.c': ['System_programming', 'Informatica', 'Security'],
        '.cs': ['Informatica', 'System_programming', 'Game_Development'],
        '.rb': ['Web_development', 'Informatica', 'System_programming'],
        '.go': ['System_programming', 'Web_development', 'Informatica'],
        '.rs': ['System_programming', 'Security', 'Informatica'],

        # Database
        '.sql': ['Database', 'Data_analysis', 'Informatica'],
        '.db': ['Database', 'Data_analysis', 'Informatica'],
        '.sqlite': ['Database', 'Data_analysis', 'Mobile_Development'],

        # Multimedia
        '.mp3': ['Comunicazione', 'Game_Development', 'Mobile_Development'],
        '.mp4': ['Comunicazione', 'Game_Development', 'Mobile_Development'],
        '.avi': ['Comunicazione', 'Game_Development', 'Informatica'],
        '.jpg': ['Comunicazione', 'Web_development', 'Game_Development'],
        '.png': ['Web_development', 'Game_Development', 'Comunicazione'],
        '.gif': ['Web_development', 'Comunicazione', 'Game_Development'],

        # CAD e Design
        '.dwg': ['Architettura', 'Informatica', 'System_programming'],
        '.dxf': ['Architettura', 'Informatica', 'System_programming'],
        '.3ds': ['Game_Development', 'Architettura', 'Informatica'],

        # Sicurezza
        '.key': ['Security', 'Informatica', 'System_programming'],
        '.pem': ['Security', 'System_programming', 'Web_development'],
        '.crt': ['Security', 'System_programming', 'Web_development'],

        # Mobile
        '.apk': ['Mobile_Development', 'Informatica', 'Security'],
        '.ipa': ['Mobile_Development', 'Informatica', 'Security'],

        # Dati
        '.csv': ['Data_analysis', 'Economia', 'Informatica'],
        '.xlsx': ['Data_analysis', 'Economia', 'Informatica'],
        '.xml': ['Data_analysis', 'Web_development', 'Informatica'],
        '.json': ['Data_analysis', 'Web_development', 'AI_ML'],

        # Default per estensioni sconosciute
        '.unknown': ['Informatica', 'Data_analysis', 'System_programming']
    }

    def comprehensive_categorize_paper(row):
        """Categorizzazione completa per tutti i tipi di file"""
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

        # Se non ci sono abbastanza categorie dal contenuto, usa l'estensione
        if len(assigned) < 3 and extension in extension_categories:
            ext_cats = extension_categories[extension]
            for ext_cat in ext_cats:
                if ext_cat not in assigned:
                    assigned.append(ext_cat)

        # Se ancora non ci sono abbastanza categorie, usa estensioni generiche
        if len(assigned) < 3:
            if extension.startswith('.'):
                # Categorizzazione intelligente basata su tipo di estensione
                if extension in ['.exe', '.dll', '.sys', '.ko']:
                    defaults = ['System_programming', 'Informatica', 'Security']
                elif extension in ['.log', '.tmp', '.bak']:
                    defaults = ['System_programming', 'Data_analysis', 'Informatica']
                elif extension in ['.conf', '.cfg', '.ini']:
                    defaults = ['System_programming', 'Informatica', 'Security']
                else:
                    defaults = ['Informatica', 'Data_analysis', 'Comunicazione']
            else:
                defaults = ['Informatica', 'Data_analysis', 'AI_ML']

            for default in defaults:
                if len(assigned) >= 3:
                    break
                if default not in assigned:
                    assigned.append(default)

        # Aggiungi connessioni interdisciplinari
        interdisciplinary = {
            'AI_ML': ['Informatica', 'Data_analysis'],
            'Medicina': ['Biologia', 'Chimica'],
            'Web_development': ['Informatica', 'Comunicazione'],
            'Mobile_Development': ['Informatica', 'Comunicazione'],
            'Security': ['Informatica', 'System_programming'],
            'Game_Development': ['Informatica', 'Comunicazione'],
            'Database': ['Informatica', 'Data_analysis']
        }

        for category in assigned[:]:
            if category in interdisciplinary:
                for related in interdisciplinary[category]:
                    if related not in assigned and len(assigned) < 5:
                        assigned.append(related)

        # Assicurati di avere almeno 3 categorie
        while len(assigned) < 3:
            remaining = [cat for cat in categories if cat not in assigned]
            if remaining:
                assigned.append(remaining[0])
            else:
                break

        return '; '.join(assigned[:5])

    # Inizializza colonna vuota
    df['category'] = ''

    # Categorizza solo i file selezionati
    print("\n🔄 Categorizzazione in corso...")
    for idx in selected_indices:
        df.at[idx, 'category'] = comprehensive_categorize_paper(df.loc[idx])

    # Salva risultato
    output_file = f'training_set_all_files_{percentage}percent.csv'
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

    # Analisi per tipo di file categorizzato
    if 'extension' in df.columns:
        print(f"\n📄 FILE CATEGORIZZATI PER ESTENSIONE:")
        print("-" * 50)
        ext_counts = categorized_df['extension'].value_counts().head(10)
        for ext, count in ext_counts.items():
            print(f"{ext:10}: {count:,} file categorizzati")

    print(f"\n📈 DISTRIBUZIONE PER CATEGORIA:")
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
            status = f"⬇️  SOTTO ({15-count} mancanti)"
        else:
            status = f"⬆️  SOPRA (+{count-20})"

        print(f"{status:20} {category:25}: {count:3d} file")

    print(f"\n🎯 RIEPILOGO:")
    print("-" * 50)
    print(f"Categorie utilizzate: {total_used}/{len(categories)}")
    print(f"Categorie nel target (15-20): {in_target}")
    print(f"File con minimo 3 categorie: {sum(1 for cat_str in categorized_df['category'] if len(cat_str.split('; ')) >= 3)}")

    return df

# Esegui la categorizzazione completa
if __name__ == "__main__":
    import sys

    # Leggi percentuale da argomenti o usa default 25%
    if len(sys.argv) > 1:
        try:
            percentage = int(sys.argv[1])
        except ValueError:
            percentage = 25
    else:
        percentage = 25

    print(f"🚀 CATEGORIZZAZIONE COMPLETA AL {percentage}%")
    print("Include TUTTI i tipi di file (PDF, HTML, immagini, codice, etc.)")
    print("=" * 60)

    # Set seed per riproducibilità
    random.seed(42)

    result = categorize_all_files_partial(percentage)
