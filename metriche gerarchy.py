import pandas as pd
from rdflib import Graph, Namespace
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



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
    # L3 la lasci come category
    df['L3'] = df['category']
    return df

def calcola_metriche(df_vere_etichette, df_predizioni):
    print("Calcolo metriche di valutazione...")
    df = pd.merge(df_vere_etichette, df_predizioni, on='filename', how='inner')

    livelli = ['L1', 'L2', 'L3']
    pred_cols = ['L1_pred', 'L2_pred', 'L3_pred_ensemble']

    risultati = {}

    for lvl, pred_col in zip(livelli, pred_cols):
        y_true = df[lvl]
        y_pred = df[pred_col]

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        risultati[f'Livello {lvl}'] = {'precision': precision, 'recall': recall, 'f1': f1}

        df['match'] = (y_true == y_pred)
        n_match = df['match'].sum()
        n_diff = len(df) - n_match

        print(f"\n{lvl}:")
        print(f"- File con categoria uguale: {n_match} (su {len(df)})")
        print(f"- File con categoria diversa: {n_diff}")
        if n_diff > 0:
            diff_rows = df.loc[~df['match'], ['filename', lvl, pred_col]]
            print(f"- File con categorizzazione diversa ({len(diff_rows)} file):")
            for _, row in diff_rows.iterrows():
                print(f"  * {row['filename']}: vero = {row[lvl]}, predetto = {row[pred_col]}")


        # Matrice di confusione - heatmap
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Categoria predetta')
        plt.ylabel('Categoria vera')
        plt.title(f'Matrice di confusione {lvl}')
        plt.tight_layout()
        plt.show()

    return risultati



def main():
    input_test_csv = 'csv simone/test_set_categorized.csv'  
    predizioni_csv = 'csv simone/predictions_on_testset_full_comparison.csv'

    df_vere = pd.read_csv(input_test_csv)
    df_pred = pd.read_csv(predizioni_csv)

    ontology_path = 'Ontology.owx'
    df_vere = prepare_hierarchies(df_vere, ontology_path)

    metriche = calcola_metriche(df_vere, df_pred)

    print("Risultati metriche:")
    for livello, valori in metriche.items():
        print(f"{livello}: Precision={valori['precision']:.3f}, Recall={valori['recall']:.3f}, F1={valori['f1']:.3f}")

if __name__ == '__main__':
    main()
