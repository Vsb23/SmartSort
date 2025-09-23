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
            
            title = metadata.get('/Title', os.path.basename(pdf_path)) if metadata else None
            # Se titolo mancante, usa nome file o altra stringa significativa
            if not title or title.strip() == '':
                print(f"Erorre con il titolo del file {os.path.basename(pdf_path)}")
                title = os.path.splitext(os.path.basename(pdf_path))[0]
            author = metadata.get('/Author', os.path.basename(pdf_path)) if metadata else None
            if "," in author:
                 author = author.replace(",", "; ") # Sostituisce le virgole con punti e virgola
            # Se autore mancante, inserisci Unknown
            if not author or author.strip() == '':
                print(f"Erorre con l'autore del file {os.path.basename(pdf_path)}")

            # Usa metadata per ottenere la data di creazione
            creation_date = metadata.get('/CreationDate') if metadata else None
            if creation_date and len(str(creation_date)) > 6:
                year = str(creation_date)[2:6]  # Estrae l'anno dalla data
            else:
                year = "Unknown"
                
            category = "Scienza"
            
            extension = os.path.splitext(os.path.basename(pdf_path))[1]
            return title,author,year,category,extension
            
    except Exception as e:
        print(f"Errore nel processare {pdf_path}: {str(e)}")
        return os.path.splitext(os.path.basename(pdf_path))[0], "Unknown", "Unknown", "Scienza"
def write_to_csv(data, output_csv):
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['titolo', 'autore', 'anno', 'category', 'extension']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            titolo, autore, anno, category, extension = entry
          

            writer.writerow({'titolo': titolo, 'autore': autore, 'anno': anno, 'category': category, 'extension': extension})


# Filtra solo i file PDF
all_data = []
for f in pdfpath:
    if f.lower().endswith('.pdf'):
        full_path = os.path.join("./References", f)
        metadata_tuple = extract_metadata_from_pdf(full_path)
        all_data.append(metadata_tuple)
    else:
        print(f"Saltando file non-PDF: {f}")
        
write_to_csv(all_data, 'output.csv')
print("File CSV scritto.")


