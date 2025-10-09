import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- IMPOSTAZIONI DI RICERCA ---
target_filename = 'predictions_on_test_set_detailed.csv'
folders_to_exclude = ['references']
found_file_path = None

print(f"🔎 Inizio ricerca ricorsiva per il file '{target_filename}'...")

# --- Logica di ricerca ricorsiva ---
for dirpath, dirnames, filenames in os.walk('.'):
    for folder in folders_to_exclude:
        if folder in dirnames:
            dirnames.remove(folder)

    if target_filename in filenames:
        found_file_path = os.path.join(dirpath, target_filename)
        print(f"✅ File trovato in: {found_file_path}")
        break

# --- Esecuzione dell'analisi solo se il file è stato trovato ---
if found_file_path:
    print("\nInizio la creazione dei grafici...")
    try:
        df = pd.read_csv(found_file_path)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

        # --- Grafico per L1_pred (CORRETTO) ---
        if 'L1_pred' in df.columns:
            plt.figure()
            l1_counts = df['L1_pred'].dropna().value_counts().sort_values(ascending=False)
            sns.barplot(x=l1_counts.values, y=l1_counts.index, hue=l1_counts.index, palette='viridis', orient='h', legend=False)
            plt.title('Distribuzione Categorie Predette - Livello 1 (L1)', fontsize=16)
            plt.xlabel('Numero di File', fontsize=12)
            plt.ylabel('Categoria', fontsize=12)
            plt.tight_layout()
            plt.savefig('distribuzione_l1_pred.png')
            print("📊 Grafico 'distribuzione_l1_pred.png' salvato.")
        
        # --- Grafico per L2_pred (CORRETTO) ---
        if 'L2_pred' in df.columns:
            plt.figure()
            l2_counts = df['L2_pred'].dropna().value_counts().sort_values(ascending=False)
            sns.barplot(x=l2_counts.values, y=l2_counts.index, hue=l2_counts.index, palette='plasma', orient='h', legend=False)
            plt.title('Distribuzione Categorie Predette - Livello 2 (L2)', fontsize=16)
            plt.xlabel('Numero di File', fontsize=12)
            plt.ylabel('Categoria', fontsize=12)
            plt.tight_layout()
            plt.savefig('distribuzione_l2_pred.png')
            print("📊 Grafico 'distribuzione_l2_pred.png' salvato.")

        # --- Grafico per L3_pred (CORRETTO) ---
        if 'L3_pred' in df.columns:
            plt.figure()
            l3_counts = df['L3_pred'].dropna().value_counts().sort_values(ascending=False)
            if len(l3_counts) > 20:
                l3_counts = l3_counts.head(20)
                title_l3 = 'Distribuzione Categorie Predette - Top 20 Livello 3 (L3)'
            else:
                title_l3 = 'Distribuzione Categorie Predette - Livello 3 (L3)'
            sns.barplot(x=l3_counts.values, y=l3_counts.index, hue=l3_counts.index, palette='magma', orient='h', legend=False)
            plt.title(title_l3, fontsize=16)
            plt.xlabel('Numero di File', fontsize=12)
            plt.ylabel('Categoria', fontsize=12)
            plt.tight_layout()
            plt.savefig('distribuzione_l3_pred.png')
            print("📊 Grafico 'distribuzione_l3_pred.png' salvato.")

    except Exception as e:
        print(f"❌ Si è verificato un errore durante la creazione dei grafici: {e}")
else:
    print(f"\n❌ ERRORE CRITICO: Ricerca completata, ma il file '{target_filename}' non è stato trovato.")