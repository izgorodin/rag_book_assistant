from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from cli import BookAssistant, progress_callback
from logger import setup_logger
from file_processor import FileProcessor
from flask_socketio import SocketIO

from src.config import FLASK_SECRET_KEY
from .websocket import socketio, emit_progress
from functools import wraps

logger = setup_logger('web')

def web_progress_callback(status: str, current: int, total: int):
    """Callback для отправки прогресса через websocket."""
    emit_progress(status, current, total)

def create_app():
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    
    app.secret_key = FLASK_SECRET_KEY  # Замените на реальный секретный ключ
    USERS = {
        'admin': 'admin1q2w3e',
        'user': 'user1q2w3e'   # Замените на реальные учетные данные
    }
    
    socketio.init_app(app)
    
    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'username' not in session:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            
            if username in USERS and USERS[username] == password:
                session['username'] = username
                return redirect(url_for('index'))
            return render_template('login.html', error='Invalid credentials')
        
        return render_template('login.html')

    @app.route('/logout')
    def logout():
        session.pop('username', None)
        return redirect(url_for('login'))

    # Инициализируем BookAssistant с веб-колбэком
    assistant = BookAssistant(progress_callback=web_progress_callback)
    file_processor = FileProcessor()
    book_data = None

    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'odt'}
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'odt'}
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    # Конфигурация
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 320 * 1024 * 1024
    
    @app.route('/', methods=['GET', 'POST'])
    @login_required
    def index():
        nonlocal book_data
        if request.method == 'POST':
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'status': 'error', 'message': 'No selected file. Allowed types: txt, pdf, doc, docx, odt'})
                
                if file and allowed_file(file.filename):
                    try:
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        
                        # Отправляем начальный статус
                        emit_progress("Starting file processing", 0, 100)
                        
                        # Загружаем и обрабатываем книгу
                        book_data = assistant.load_and_process_book(file_path)
                        
                        # Отправляем финальный статус
                        emit_progress("Processing complete", 100, 100, {
                            'chunks_count': len(book_data.get_chunks())
                        })
                        
                        return jsonify({
                            'status': 'success',
                            'message': 'File processed successfully',
                            'chunks_count': len(book_data.get_chunks())
                        })
                    except Exception as e:
                        emit_progress("Error", 0, 100, {'error': str(e)})
                        return jsonify({'status': 'error', 'message': str(e)})
                else:
                    return jsonify({'status': 'error', 'message': 'Invalid file type'})
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
