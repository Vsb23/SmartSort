import os
import PyPDF2
import csv

def extract_metadata_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = pdf_reader.metadata
            
            title = metadata.get('/Title', None) if metadata else None
            if not title or title.strip() == '':
                title = os.path.splitext(os.path.basename(pdf_path))[0]

            filename= os.path.basename(pdf_path) 

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
            print(title, filename, author, year, category, extension)
            return title, filename, author, year, category, extension, "file"
    except Exception as e:
        # In caso di errore (es. file non leggibile come PDF)
        name_no_ext = os.path.splitext(os.path.basename(pdf_path))[0]
        extension = os.path.splitext(pdf_path)[1].lower()
        return name_no_ext, "Unknown", "Unknown", "Unknown", extension, "file"

def explore_folder_recursive(base_path):
    all_data = []
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            full_path = os.path.join(root, d)
            relative_path = os.path.relpath(full_path, base_path)
            all_data.append((d, "", "Unknown", "Unknown", "Unknown", "", "folder", relative_path))
        for f in files:
            full_path = os.path.join(root, f)
            containing_folder = os.path.dirname(full_path)
            relative_folder = os.path.relpath(containing_folder, base_path)
            metadata_tuple = extract_metadata_from_pdf(full_path)
            all_data.append(metadata_tuple + (relative_folder,))
    return all_data



def write_to_csv(data, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['titolo','filename', 'autore', 'anno', 'category', 'extension', 'type', 'relative_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            titolo, filename, autore, anno, categoria, estensione, tipo, relpath = entry
            writer.writerow({
                'titolo': titolo,
                'filename': filename,
                'autore': autore,
                'anno': anno,
                'category': categoria,
                'extension': estensione,
                'type': tipo,
                'relative_path': relpath
            })

# Percorso base da esplorare ricorsivamente
base_folder = "./References"
all_data = explore_folder_recursive(base_folder)

write_to_csv(all_data, 'output.csv')
print("File CSV scritto con struttura ricorsiva.")
