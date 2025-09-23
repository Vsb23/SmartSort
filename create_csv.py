import os
import PyPDF2
import csv

pdfpath = os.listdir("./References")

def extract_metadata_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            # Per PyPDF2 versione 2.0+, usa PdfReader invece di PdfFileReader
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = pdf_reader.metadata
            
            title = metadata.get('/Title', os.path.basename(pdf_path)) if metadata else os.path.basename(pdf_path)
            author = metadata.get('/Author', 'Unknown') if metadata else "Unknown"
            if "," in author:
                 author = author.replace(",", "; ") # Sostituisce le virgole con punti e virgola

            # Usa metadata per ottenere la data di creazione
            creation_date = metadata.get('/CreationDate') if metadata else None
            if creation_date and len(str(creation_date)) > 6:
                year = str(creation_date)[2:6]  # Estrae l'anno dalla data
            else:
                year = "Unknown"
                
            category = "Scienza"
            

            return title,author,year,category
            
    except Exception as e:
        print(f"Errore nel processare {pdf_path}: {str(e)}")
        return os.path.basename(pdf_path), "Unknown", "Unknown", "Unknown"
def write_to_csv(data, output_csv):
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['titolo', 'autore', 'anno', 'category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            titolo, autore, anno, category = entry.split(',')
            if not titolo or not autore or not anno or not category:
                print(f"Skipping incomplete entry: {entry}")
                continue
            writer.writerow({'titolo': titolo, 'autore': autore, 'anno': anno, 'category': category})


# Filtra solo i file PDF
for f in pdfpath:
    if f.lower().endswith('.pdf'):
        full_path = os.path.join("./References", f)
        print(extract_metadata_from_pdf(full_path))
        titolo, autore, anno, category = extract_metadata_from_pdf(full_path)
        data = [f"{titolo},{autore},{anno},{category}"]
        write_to_csv(data, 'output.csv')
        print(f"Metadata estratti per {f} e scritti in output.csv")
    else:
        print(f"Saltando file non-PDF: {f}")

