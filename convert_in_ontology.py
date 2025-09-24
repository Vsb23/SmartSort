import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal, RDF

# URI base del tuo progetto
BASE = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")

g = Graph()
g.bind("icon", BASE)

df = pd.read_csv("output.csv")

for idx, row in df.iterrows():
    doc_id = f"document_{idx}"
    doc_uri = BASE[doc_id]

    g.add((doc_uri, RDF.type, BASE.Documento))
    g.add((doc_uri, BASE.hasTitle, Literal(row["titolo"])))
    g.add((doc_uri, BASE.hasAuthorName, Literal(row["autore"])))
    g.add((doc_uri, BASE.hasPublicationYear, Literal(row["anno"])))
    g.add((doc_uri, BASE.belongsToCategory, Literal(row["category"])))
    g.add((doc_uri, BASE.path, Literal(row["relative_path"])))
    g.add((doc_uri, BASE.extension, Literal(row["extension"])))

# Salva il grafo in Turtle
g.serialize("istanze.owl", format="turtle")
