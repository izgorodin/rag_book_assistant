import logging
import traceback
import colorlog
import os
from datetime import datetime
import json

class LoggerManager:
    _instance = None
    _loggers = {}
    
    def __init__(self):
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Основной лог файл для приложения
        self.setup_main_logger()
        
        # Лог файл для результатов RAG
        self.setup_rag_logger()
    
    def setup_main_logger(self):
        logger = logging.getLogger('main')
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            
            # Файловый хендлер с JSON форматированием
            file_handler = logging.FileHandler(
                os.path.join(self.log_dir, 'app.log')
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JSONFormatter())
            
            # Консольный хендлер остается читаемым для человека
            console_handler = colorlog.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)s [%(name)s] %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            ))
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        self._loggers['main'] = logger
    
    def setup_rag_logger(self):
        logger = logging.getLogger('rag')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
            file_handler = logging.FileHandler(
                os.path.join(self.log_dir, 'rag_results.log')
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            
            logger.addHandler(file_handler)
            
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
