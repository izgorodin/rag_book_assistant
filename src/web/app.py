from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from src.cli import BookAssistant
from src.utils.logger import get_main_logger, get_rag_logger
from src.file_processor import FileProcessor
from src.config import FLASK_SECRET_KEY
from src.web.websocket import emit_progress, socketio
from functools import wraps

# Initialize loggers for the application
logger = get_main_logger()
rag_logger = get_rag_logger()

def web_progress_callback(status: str, current: int, total: int):
    """Callback function to send progress updates through WebSocket."""
    emit_progress(status, current, total)  # Emit progress status
    logger.debug(f"Progress update: {status} ({current}/{total})")  # Log progress update

def create_app(init_services=True):
    """Create and configure the Flask application."""
    # Define upload folder path and create it if it doesn't exist
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Upload folder initialized: {UPLOAD_FOLDER}")
    
    # Define user credentials
    USERS = {
        'admin': os.environ.get('ADMIN_PASSWORD', 'admin1q2w3e'),
        'tester1': os.environ.get('TESTER_PASSWORD', '41dsf3qw7sDa')
    }
    
    # Initialize the Flask application
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'),
                static_url_path='')
    
    # Application configuration settings
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 320 * 1024 * 1024  # Set max upload size
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', FLASK_SECRET_KEY)  # Set secret key
    
    # Initialize services if required
    if init_services:
        assistant = BookAssistant(progress_callback=web_progress_callback)  # Create BookAssistant with progress callback
        logger.info("BookAssistant initialized with web progress callback")
    else:
        assistant = None
        logger.info("BookAssistant initialization skipped")
        
    book_data = None  # Initialize book data variable
    
    socketio.init_app(app)  # Initialize SocketIO with the Flask app
    logger.info("SocketIO initialized")
    
    def login_required(f):
        """Decorator to ensure user is logged in before accessing certain routes."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'username' not in session:  # Check if user is logged in
                logger.debug("Unauthorized access attempt, redirecting to login")
                return redirect(url_for('login'))  # Redirect to login if not logged in
            return f(*args, **kwargs)  # Proceed to the requested function
        return decorated_function

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Handle user login."""
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            
            # Check if credentials are valid
            if username in USERS and USERS[username] == password:
                session['username'] = username  # Store username in session
                logger.info(f"User '{username}' logged in successfully")
                return redirect(url_for('index'))  # Redirect to index page
            logger.warning(f"Failed login attempt for user: {username}")
            return render_template('login.html', error='Invalid credentials')  # Render login page with error
        
        return render_template('login.html')  # Render login page for GET request

    @app.route('/logout')
    def logout():
        """Handle user logout."""
        username = session.pop('username', None)  # Remove username from session
        if username:
            logger.info(f"User '{username}' logged out")  # Log logout event
        return redirect(url_for('login'))  # Redirect to login page

    # Initialize BookAssistant with web callback
    assistant = BookAssistant(progress_callback=web_progress_callback)
    file_processor = FileProcessor()  # Create FileProcessor instance
    book_data = None  # Initialize book data variable

    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'odt'}  # Define allowed file extensions
    
    def allowed_file(filename):
        """Check if the uploaded file has an allowed extension."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/', methods=['GET', 'POST'])
    @login_required
    def index():
        """Render the main index page and handle file uploads and questions."""
        nonlocal book_data  # Use nonlocal variable for book data
        if request.method == 'POST':
            if 'file' in request.files:  # Check if a file is uploaded
                file = request.files['file']
                if file.filename == '':
                    logger.warning("Empty file upload attempt")
                    return jsonify({'status': 'error', 'message': 'No selected file. Allowed types: txt, pdf, doc, docx, odt'})
                
                if file and allowed_file(file.filename):  # Validate file type
                    try:
                        filename = secure_filename(file.filename)  # Secure the filename
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Create file path
                        file.save(file_path)  # Save the uploaded file
                        logger.info(f"File uploaded: {filename}")
                        
                        # Emit initial processing status
                        emit_progress("Starting file processing", 0, 100)
                        
                        # Load and process the book
                        book_data = assistant.load_and_process_book(file_path)
                        logger.info(f"File processed: {filename}")
                        
                        # Emit final processing status
                        emit_progress("Processing complete", 100, 100, {
                            'chunks_count': len(book_data.get_chunks())  # Include chunk count in progress
                        })
                        
                        return jsonify({
                            'status': 'success',
                            'message': 'File processed successfully',
                            'chunks_count': len(book_data.get_chunks())  # Return chunk count
                        })
                    except Exception as e:
                        logger.error(f"Error processing file: {str(e)}")
                        emit_progress("Error", 0, 100, {'error': str(e)})  # Emit error status
                        return jsonify({'status': 'error', 'message': str(e)})  # Return error message
                else:
                    logger.warning(f"Invalid file type: {file.filename}")
                    return jsonify({'status': 'error', 'message': 'Invalid file type'})  # Return invalid file type message
            elif 'question' in request.form:  # Check if a question is asked
                if not book_data:
                    logger.warning("Question asked without loaded book")
                    return jsonify({'status': 'error', 'message': 'No book loaded'})  # Return error if no book is loaded
                    
                try:
                    question = request.form['question']  # Get the question from the form
                    logger.info(f"Processing question: {question}")
                    answer = assistant.answer_question(request.form['question'], book_data)  # Get the answer
                    logger.info("Answer generated successfully")
                    return jsonify({'status': 'success', 'answer': answer})  # Return the answer
                except Exception as e:
                    logger.error(f"Error answering question: {str(e)}")
                    return jsonify({'status': 'error', 'message': str(e)})  # Return error message
                    
        return render_template('index.html')  # Render index page for GET request

    @app.route('/check_book_loaded', methods=['GET'])
    def check_book_loaded():
        """Check if a book is loaded and return its status."""
        status = book_data is not None  # Check if book data is available
        logger.debug(f"Book load status checked: {status}")
        return jsonify({'book_loaded': status})  # Return book load status

    @app.route('/ask', methods=['POST'])
    def ask():
        """Handle question asking and return the answer."""
        nonlocal book_data  # Use nonlocal variable for book data
        if not book_data:
            logger.warning("Question asked without loaded book")
            return jsonify({'status': 'error', 'message': 'No book loaded'})  # Return error if no book is loaded
            
        question = request.form.get('question')  # Get the question from the form
        if not question:
            logger.warning("Empty question received")
            return jsonify({'status': 'error', 'message': 'No question provided'})  # Return error if no question is provided
            
        try:
            logger.info(f"Processing question: {question}")
            answer = assistant.answer_question(question, book_data)  # Get the answer
            logger.info("Answer generated successfully")
            return jsonify({'status': 'success', 'answer': answer})  # Return the answer
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)})  # Return error message

    return app  # Return the configured Flask app

def run_web_app():
    """Start the web application."""
    logger.info("Starting web application")
    app = create_app()  # Create the Flask app
    socketio.init_app(app)  # Initialize SocketIO with the app
    socketio.run(app, debug=True)  # Run the app with debug mode

if __name__ == "__main__":
    run_web_app()  # Execute the web application
