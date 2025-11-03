import os
import PyPDF2
import csv

def extract_metadata_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = pdf_reader.metadata
            title = os.path.splitext(os.path.basename(pdf_path))[0]
            filename = os.path.basename(pdf_path)
            author = metadata.get('/Author', "") if metadata else ""
            if author is None: author = ""
            if "," in author: author = author.replace(",", "; ")
            creation_date = metadata.get('/CreationDate') if metadata else None
            year = str(creation_date)[2:6] if creation_date and len(str(creation_date)) > 6 else ""
            extension = os.path.splitext(pdf_path)[1].lower()
            return title, filename, author, year, "", extension, "file"
    except Exception as e:
        print(f"⚠️  Impossibile leggere i metadati da '{pdf_path}'. Uso fallback.")
        name_no_ext = os.path.splitext(os.path.basename(pdf_path))[0]
        extension = os.path.splitext(pdf_path)[1].lower()
        return name_no_ext, os.path.basename(pdf_path), "", "", "", extension, "file"

def explore_folder_recursive(base_path):
    all_data = []
    for root, dirs, files in os.walk(base_path):
        # Processa prima le cartelle
        for d in dirs:
            full_path = os.path.join(root, d)
            relative_path = os.path.relpath(full_path, base_path)
            # Aggiungi info cartella: (nome cartella, nome cartella, "", "", "", "", "folder", percorso relativo)
            all_data.append((d, d, "", "", "", "", "folder", relative_path))
        # Poi processa i file PDF
        for f in files:
            if f.lower().endswith('.pdf'):
                full_path = os.path.join(root, f)
                containing_folder = os.path.dirname(full_path)
                relative_folder = os.path.relpath(containing_folder, base_path)
                metadata_tuple = extract_metadata_from_pdf(full_path)
                all_data.append(metadata_tuple + (relative_folder,))
    return all_data

def write_to_csv(data, output_csv):
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['titolo','filename', 'autore', 'anno', 'category', 'extension', 'type', 'relative_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            if len(entry) == 8:
                titolo, filename, autore, anno, categoria, estensione, tipo, relpath = entry
                writer.writerow({'titolo': titolo, 'filename': filename, 'autore': autore, 'anno': anno, 'category': categoria, 'extension': estensione, 'type': tipo, 'relative_path': relpath})
            else:
                 print(f"⚠️ Attenzione: Record saltato a causa di un numero errato di campi: {entry}")

def process_single_task(source_folder, output_csv):
    """
    Processa una singola cartella (task) e scrive i suoi contenuti in un CSV.
    Restituisce il numero di file e record.
    """
    print(f"\n--- Inizio processo per: '{source_folder}' -> '{output_csv}' ---")
    if not os.path.isdir(source_folder):
        print(f"❌ ERRORE: La cartella sorgente '{source_folder}' non esiste. Task saltato.")
        print("-" * 70)
        return 0, 0

    all_data = explore_folder_recursive(source_folder)
    write_to_csv(all_data, output_csv)

    print(f"✅ File CSV '{output_csv}' creato con successo.")

    num_files = sum(1 for entry in all_data if len(entry) == 8 and entry[6] == "file")
    total_records = len(all_data)

    print(f"Numero di file PDF trovati: {num_files}")
    print(f"Numero totale di record (file + folder): {total_records}")
    print("-" * 70)

    return num_files, total_records

def process_folder_to_csv(source_folder, output_csv):
    print(f"--- Inizio processo per la cartella: '{source_folder}' ---")
    if not os.path.isdir(source_folder):
        print(f"❌ ERRORE: La cartella '{source_folder}' non esiste. Processo saltato.")
        print("-" * 50)
        return 0, 0  # Restituisce 0 se la cartella non esiste

    all_data = explore_folder_recursive(source_folder)
    write_to_csv(all_data, output_csv)
    
    print(f"✅ File CSV '{output_csv}' creato con successo.")
    
    num_files = sum(1 for entry in all_data if entry[6] == "file")
    total_records = len(all_data)
    
    print(f"Numero di file PDF trovati: {num_files}")
    print(f"Numero totale di record (file + folder): {total_records}")
    print("-" * 50)
    
    return num_files, total_records # Restituisce i valori calcolati

if __name__ == "__main__":
    os.makedirs("./training_result", exist_ok=True)
    os.makedirs("./test_result", exist_ok=True)
    os.makedirs("./test_result_2", exist_ok=True)
    os.makedirs("./test_result_3", exist_ok=True)
    total_files_processed = 0
    total_records_processed = 0

    # 1. Processa la cartella di Training e aggiorna i totali
    num_f, num_r = process_folder_to_csv("training_data", "./training_result/output.csv")
    total_files_processed += num_f
    total_records_processed += num_r
    
    # 2. Processa la cartella di Test e aggiorna i totali
    num_f, num_r = process_folder_to_csv("test_data", "./test_result/test_output.csv")
    total_files_processed += num_f
    total_records_processed += num_r
    
    num_f, num_r = process_folder_to_csv("test_data_2", "./test_result_2/test_output_2.csv")
    total_files_processed += num_f
    total_records_processed += num_r

    num_f, num_r = process_folder_to_csv("test_data_3", "./test_result_3/test_output_3.csv")
    total_files_processed += num_f
    total_records_processed += num_r

    print("🎉 Tutti i processi sono stati completati.")
    print("\n" + "--- RIEPILOGO COMPLESSIVO ---".center(50))
    print(f"Numero totale di file PDF processati: {total_files_processed}")
    print(f"Numero totale di record (file + folder): {total_records_processed}")
    print("-" * 50)