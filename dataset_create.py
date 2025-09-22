import csv

def create_instance(class_name, instance_id, properties):
    instance = f'<owl:NamedIndividual rdf:about="#{instance_id}">\n'
    instance += f'  <rdf:type rdf:resource="#{class_name}"/>\n'
    for prop, val in properties.items():
        if isinstance(val, list):
            for v in val:
                instance += f'  <{prop} rdf:resource="#{v}"/>\n'
        elif isinstance(val, str) and val.startswith('#'):
            instance += f'  <{prop} rdf:resource="{val}"/>\n'
        else:
            instance += f'  <{prop}>{val}</{prop}>\n'
    instance += '</owl:NamedIndividual>\n'
    return instance

# Esempio di CSV con colonne: id,titolo,anno,autore,category
csv_file = 'data.csv'

rdf_instances = ''
with open(csv_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        instance_id = row['id']
        properties = {
            'hasTitle': row['titolo'],
            'hasPublicationYear': row['anno'],
            'hasAuthor': f"#{row['autore']}",
            'belongsToCategory': f"#{row['category']}"
        }
        rdf_instances += create_instance('Documento', instance_id, properties)

owl_content = f'''<rdf:RDF xmlns="http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <owl:Ontology rdf:about="http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11"/>
{rdf_instances}</rdf:RDF>'''

with open('output.owl', 'w', encoding='utf-8') as f:
    f.write(owl_content)

print('File OWL generato: output.owl')
