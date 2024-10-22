import os
from app import extract_text, UPLOAD_FOLDER
from src.text_processing import load_and_preprocess_text

def main():
    print("Starting the script...")
    
    # Используйте абсолютный путь к файлу
    pdf_filename = 'voina-i-mir.pdf'  # Замените на реальное имя файла
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
    
    print(f"Attempting to process file: {pdf_path}")
    
    # Проверка существования файла
    if not os.path.exists(pdf_path):
        print(f"Error: File does not exist at {pdf_path}")
        return

    try:
        print("Extracting text from PDF...")
        extracted_text = extract_text(pdf_path)
        
        print("Preprocessing extracted text...")
        processed_data = load_and_preprocess_text(extracted_text)
        
        print("\nExtraction and preprocessing completed. Results:")
        print(f"Text length: {len(processed_data['text'])}")
        print(f"Dates found: {len(processed_data['dates'])}")
        print(f"Entities found: {len(processed_data['entities'])}")
        print(f"Key phrases found: {len(processed_data['key_phrases'])}")
        
        # Вывод первых 500 символов текста
        print("\nFirst 500 characters of extracted text:")
        print(processed_data['text'][:500])
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
