import os
import logging
from flask import Flask, render_template, request, jsonify
from src.cli import load_and_process_book, answer_question
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем директорию uploads, если она не существует
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 320 * 1024 * 1024  # Увеличиваем до 320 MB

ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Глобальные переменные для хранения данных книги
book_data = {
    'chunks': None,
    'embeddings': None,
    'loaded': False
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    global book_data
    
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                logger.info(f"File saved: {file_path}")
                
                try:
                    book_data['chunks'], book_data['embeddings'] = load_and_process_book(file_path)
                    book_data['loaded'] = True
                    logger.info(f"Book processed. Chunks: {len(book_data['chunks'])}, Embeddings: {len(book_data['embeddings'])}")
                    return jsonify({"success": True, "message": "File uploaded and processed successfully"})
                except Exception as e:
                    logger.error(f"Error processing file: {str(e)}")
                    return jsonify({"success": False, "message": f"Error processing file: {str(e)}"}), 500
                finally:
                    # Удаляем файл после обработки
                    os.remove(file_path)
            else:
                return jsonify({"success": False, "message": "Invalid file type"}), 400
        elif 'question' in request.form:
            question = request.form['question']
            if book_data['loaded']:
                try:
                    answer = answer_question(question, book_data['chunks'], book_data['embeddings'])
                    logger.info(f"Generated answer for question: {question}")
                    return jsonify({"success": True, "answer": answer})
                except Exception as e:
                    logger.error(f"Error generating answer: {str(e)}")
                    return jsonify({"success": False, "message": f"Error generating answer: {str(e)}"}), 500
            else:
                return jsonify({"success": False, "message": "No book loaded. Please upload a book first."}), 400
    
    return render_template('index.html', book_loaded=book_data['loaded'])

@app.route('/check_book_loaded', methods=['GET'])
def check_book_loaded():
    return jsonify({"book_loaded": book_data['loaded']})

if __name__ == '__main__':
    app.run(debug=True)
