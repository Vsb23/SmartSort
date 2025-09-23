import os
import PyPDF2
import csv

pdfpath = os.listdir("./References")

def extract_metadata_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = pdf_reader.metadata
            
            title = metadata.get('/Title', None) if metadata else None
            if not title or title.strip() == '':
                title = os.path.splitext(os.path.basename(pdf_path))[0]
            
            author = metadata.get('/Author', None) if metadata else None
            if not author or author.strip() == '':
                author = "Unknown"
            if "," in author:
                author = author.replace(",", "; ")
            
            creation_date = metadata.get('/CreationDate') if metadata else None
            if creation_date and len(str(creation_date)) > 6:
                year = str(creation_date)[2:6]
            else:
                year = "Unknown"
            
            category = "Scienza"
            extension = os.path.splitext(pdf_path)[1].lower()
            
            return title, author, year, category, extension
    except Exception as e:
        # Se non è PDF o errore, ritorna valori default
        name_no_ext = os.path.splitext(os.path.basename(pdf_path))[0]
        extension = os.path.splitext(pdf_path)[1].lower()
        return name_no_ext, "Unknown", "Unknown", "Unknown", extension

def write_to_csv(data, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['titolo', 'autore', 'anno', 'category', 'extension']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            titolo, autore, anno, categoria, estensione = entry
            writer.writerow({'titolo': titolo, 'autore': autore, 'anno': anno, 'category': categoria, 'extension': estensione})

all_data = []
for f in pdfpath:
    full_path = os.path.join("./References", f)
    metadata_tuple = extract_metadata_from_pdf(full_path)
    all_data.append(metadata_tuple)

write_to_csv(all_data, 'output.csv')
print("File CSV scritto.")
