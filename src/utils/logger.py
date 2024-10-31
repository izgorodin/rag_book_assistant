import logging
from logging.handlers import RotatingFileHandler
import traceback
import colorlog
import os
from datetime import datetime
import json

class LineCountRotatingHandler(RotatingFileHandler):
    """Custom handler that rotates logs based on line count"""
    def __init__(self, filename, max_lines=10000, **kwargs):
        super().__init__(filename, **kwargs)
        self.max_lines = max_lines
        
    def shouldRollover(self, record):
        if self.stream is None:
            self.stream = self._open()
            
        try:
            self.stream.seek(0)
            line_count = sum(1 for _ in self.stream)
            self.stream.seek(0, 2)  # Return to end of file
            return line_count >= self.max_lines
        except Exception:
            return False

class LoggerManager:
    _instance = None
    _loggers = {}
    
    def __init__(self):
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # JSON форматтер для файла
        self.file_handler = LineCountRotatingHandler(
            os.path.join(self.log_dir, 'app.log'),
            max_lines=10000,
            maxBytes=0,
            backupCount=1
        )
        self.file_handler.setFormatter(JSONFormatter())
        
        # Человекочитаемый форматтер для консоли
        self.console_handler = colorlog.StreamHandler()
        self.console_handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)s: %(message)s%(reset)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        
        self.setup_main_logger()
        self.setup_rag_logger()
    
    def setup_main_logger(self):
        logger = logging.getLogger('main')
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            logger.addHandler(self.file_handler)
            logger.addHandler(self.console_handler)
        self._loggers['main'] = logger
    
    def setup_rag_logger(self):
        logger = logging.getLogger('rag')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            logger.addHandler(self.file_handler)
            logger.addHandler(self.console_handler)
        self._loggers['rag'] = logger
    
    @classmethod
    def get_logger(cls, name='main'):
        if cls._instance is None:
            cls._instance = LoggerManager()
        return cls._instance._loggers.get(name)

def get_main_logger():
    return LoggerManager.get_logger('main')

def get_rag_logger():
    return LoggerManager.get_logger('rag')

def get_structured_logger(name: str):
    logger = logging.getLogger(name)
    
    def structured_log(msg, **kwargs):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "service": name,
            **kwargs
        }
        return json.dumps(log_data)
        
    return logger

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
            
        if record.exc_info:
            log_data['traceback'] = traceback.format_exception(*record.exc_info)
            
        return json.dumps(log_data)

def setup_structured_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger
