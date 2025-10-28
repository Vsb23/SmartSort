import pandas as pd
from rdflib import Graph, Namespace
from sklearn.metrics import precision_recall_fscore_support



def get_parents(g, ns, child_class_name):
    parents = set()
    if child_class_name == "Altro" or not child_class_name:
        return parents
    child_uri = ns[child_class_name]
    from rdflib.namespace import RDFS
    for s, p, o in g.triples((child_uri, RDFS.subClassOf, None)):
        if '#' in o:
            parents.add(o.split('#')[-1])
    return parents

def get_hierarchy_levels(g, ns, child_class_name):
    l2 = None
    l1 = None
    parents = get_parents(g, ns, child_class_name)
    if parents:
        l2 = next(iter(parents))
        grand_parents = get_parents(g, ns, l2)
        if grand_parents:
            l1 = next(iter(grand_parents))
    return l1, l2

def prepare_hierarchies(df, ontology_path):
    g = Graph()
    g.parse(ontology_path, format='xml')
    ns = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")

    l1_list = []
    l2_list = []

    for child_class in df['category']:
        l1, l2 = get_hierarchy_levels(g, ns, child_class)
        l1_list.append(l1 if l1 else "Altro")
        l2_list.append(l2 if l2 else "Altro")

    df['L1'] = l1_list
    df['L2'] = l2_list
    df['L3'] = df['category']
    return df

def calcola_metriche(df_vere_etichette, df_predizioni, test_set_name):
    """
    Calcola le metriche PRF1 pesate senza generare matrici di confusione 
    o report dettagliati per aderire alla linea guida.
    """
    print(f"Calcolo metriche di valutazione per {test_set_name}...")
    df = pd.merge(df_vere_etichette, df_predizioni, on='filename', how='inner')

    livelli = ['L1', 'L2', 'L3']
    pred_cols = ['L1_pred', 'L2_pred', 'L3_pred_ensemble']

    risultati = {}
    
    # Rimosso il setup della cartella di output (non ci sono più immagini)

    for lvl, pred_col in zip(livelli, pred_cols):
        y_true = df[lvl]
        y_pred = df[pred_col]

        # Calcolo PRF1 Pesato
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        risultati[f'Livello {lvl}'] = {'precision': precision, 'recall': recall, 'f1': f1}

        df['match'] = (y_true == y_pred)
        n_match = df['match'].sum()
        
        # Stampa riassuntiva pulita (solo Accuratezza)
        print(f"\n{lvl} (Accuratezza Esemble):")
        print(f"- Accuratezza: {n_match / len(df):.4f} (su {len(df)} file)")
        
        # Rimosso: Matrice di confusione e report dettagliato dei file errati.

    return risultati

def process_test_set_evaluation(input_csv, predizioni_csv, ontology_path, test_set_name):
    """Carica i dati e le predizioni, prepara la gerarchia e calcola le metriche per un singolo set."""
    try:
        df_vere = pd.read_csv(input_csv)
        df_pred = pd.read_csv(predizioni_csv) 
    except FileNotFoundError as e:
        print(f"❌ ERRORE: File non trovato per {test_set_name}: {e}")
        return

    df_vere = prepare_hierarchies(df_vere, ontology_path)

    metriche = calcola_metriche(df_vere, df_pred, test_set_name) 

    print(f"\n--- Risultati metriche {test_set_name} (PRF1 Pesato) ---")
    for livello, valori in metriche.items():
        print(f"{livello}: Precision={valori['precision']:.3f}, Recall={valori['recall']:.3f}, F1={valori['f1']:.3f}")
    print("-" * 50)


def main():
    ontology_path = 'Ontology.owx'
    
    # Configurazione Test Set 1
    config_test_1 = {
        'name': 'Test_Set_1',
        'input_csv': 'test_result/test_set_categorized.csv',
        'predizioni_csv': 'test_result/predictions_on_testset_full_comparison_gerarchy.csv'
    }

    # Configurazione Test Set 2
    config_test_2 = {
        'name': 'Test_Set_2',
        'input_csv': 'test_result_2/test_set_2_categorized.csv',
        'predizioni_csv': 'test_result_2/predictions_on_testset_2_full_comparison_gerarchy.csv'
    }

    print("=========================================")
    process_test_set_evaluation(
        config_test_1['input_csv'], 
        config_test_1['predizioni_csv'], 
        ontology_path, 
        config_test_1['name']
    )
    
    process_test_set_evaluation(
        config_test_2['input_csv'], 
        config_test_2['predizioni_csv'], 
        ontology_path, 
        config_test_2['name']
    )
    print("🎉 Valutazione completata per tutti i set!")


if __name__ == '__main__':
    main()