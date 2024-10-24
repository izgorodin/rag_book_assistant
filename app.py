import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from src.cli import load_and_process_book, answer_question
from src.file_processor import FileProcessor
from src.rag import rag_query
from src.logger import setup_logger  # Import the setup_logger function


app = Flask(__name__)

# Initialize the logger using setup_logger from logger.py
logger = setup_logger('app.log')

# Create the uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 320 * 1024 * 1024  # Increase to 320 MB

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'odt'}

# Global variable to store book data
book_data = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    logger.info("Index route accessed")
    if request.method == 'POST':
        logger.info("POST request received")
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({'status': 'error', 'message': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'status': 'error', 'message': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                global book_data
                file_processor = FileProcessor()
                text_content = file_processor.process_file(file_path)
                book_data = load_and_process_book(text_content)
                logger.info(f"Book data loaded: {len(book_data.chunks)} chunks")
                return jsonify({'status': 'success', 'message': 'File uploaded and processed successfully'})
            except Exception as e:
                logger.exception(f"Error processing file: {str(e)}")
                return jsonify({'status': 'error', 'message': f'Error processing file: {str(e)}'})
        else:
            return jsonify({'status': 'error', 'message': 'File type not allowed'})
    logger.info("Rendering index.html")
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    global book_data
    if book_data is None:
        return jsonify({'status': 'error', 'message': 'No book data loaded'})
    
    query = request.json.get('question')
    if not query:
        return jsonify({'status': 'error', 'message': 'No question provided'})
    
    try:
        answer = answer_question(query, book_data)
        return jsonify({'status': 'success', 'answer': answer})
    except Exception as e:
        logger.exception(f"Error answering question: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error answering question: {str(e)}'})

@app.route('/check_book_loaded', methods=['GET'])
def check_book_loaded():
    return jsonify({"book_loaded": book_data is not None})

@app.route('/debug', methods=['GET'])
def debug_info():
    return jsonify({
        'book_data_loaded': book_data is not None,
        'chunks_count': len(book_data.chunks) if book_data else 0,
        'embeddings_count': len(book_data.embeddings) if book_data else 0,
    })

if __name__ == '__main__':
    port = 5001
    while True:
        try:
            logger.info(f"Starting Flask app on port {port}")
            app.run(debug=True, host='0.0.0.0', port=port)
            break
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"Port {port} is in use, trying the next one.")
                port += 1
            else:
                logger.exception("Failed to start the Flask app")
                raise
