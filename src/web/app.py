from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from src.cli import BookAssistant
from src.logger import setup_logger

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
                return handle_file_upload(request.files['file'], assistant)
            elif 'question' in request.form:
                return handle_question(request.form['question'], book_data, assistant)
        return render_template('index.html')

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

