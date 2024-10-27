from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from cli import BookAssistant
from logger import setup_logger

logger = setup_logger('web')

def create_app():
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    
    assistant = BookAssistant()
    book_data = None

    # Конфигурация
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 320 * 1024 * 1024
    
    @app.route('/', methods=['GET', 'POST'])
    def index():
        nonlocal book_data
        if request.method == 'POST':
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'status': 'error', 'message': 'No selected file'})
                
                if file:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    try:
                        book_data = assistant.load_and_process_book(file_path)
                        return jsonify({
                            'status': 'success',
                            'message': 'File processed successfully',
                            'chunks_count': len(book_data.get_chunks())
                        })
                    except Exception as e:
                        logger.error(f"Error processing file: {str(e)}")
                        return jsonify({'status': 'error', 'message': str(e)})
            elif 'question' in request.form:
                if not book_data:
                    return jsonify({'status': 'error', 'message': 'No book loaded'})
                    
                try:
                    answer = assistant.answer_question(request.form['question'], book_data)
                    return jsonify({'status': 'success', 'answer': answer})
                except Exception as e:
                    logger.error(f"Error answering question: {str(e)}")
                    return jsonify({'status': 'error', 'message': str(e)})
                    
        return render_template('index.html')

    @app.route('/check_book_loaded', methods=['GET'])
    def check_book_loaded():
        return jsonify({'book_loaded': book_data is not None})

    @app.route('/ask', methods=['POST'])
    def ask():
        nonlocal book_data
        if not book_data:
            return jsonify({'status': 'error', 'message': 'No book loaded'})
            
        question = request.form.get('question')
        if not question:
            return jsonify({'status': 'error', 'message': 'No question provided'})
            
        try:
            answer = assistant.answer_question(question, book_data)
            return jsonify({'status': 'success', 'answer': answer})
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)})

    return app

def run_web_app(host='0.0.0.0', port=5001):
    app = create_app()
    while True:
        try:
            logger.info(f"Starting Flask app on port {port}")
            app.run(debug=True, host=host, port=port)
            break
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"Port {port} is in use, trying {port + 1}")
                port += 1
            else:
                raise
