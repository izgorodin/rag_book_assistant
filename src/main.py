from openai import OpenAI
from src.cli import BookAssistant
from src.cache_manager import CacheManager
from src.utils.logger import get_main_logger
from src.config import CACHE_DIR
import sys
import os

logger = get_main_logger()

def main():
    """Main entry point for the application."""
    try:
        # Создаем директорию для кэша, если её нет
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Инициализируем CacheManager с указанием директории
        cache_manager = CacheManager(cache_dir=CACHE_DIR)
        logger.info(f"CacheManager initialized at {CACHE_DIR}")

        # Инициализируем OpenAI клиент
        openai_client = OpenAI()

        # Создаем экземпляр BookAssistant с необходимыми зависимостями
        assistant = BookAssistant(
            openai_client=openai_client,
            cache_manager=cache_manager
        )

        # Запускаем CLI режим
        if len(sys.argv) > 1 and sys.argv[1] == "cli":
            assistant.run()
        else:
            logger.error("Please specify the mode: python -m src.main cli")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()