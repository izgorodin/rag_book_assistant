# Базовый образ
FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание и активация виртуального окружения
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

# Установка пакета в режиме разработки
RUN pip install -e .

# Создание необходимых директорий
RUN mkdir -p uploads logs data

# Установка NLTK данных
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords wordnet words averaged_perceptron_tagger omw-1.4 tagsets maxent_ne_chunker universal_tagset

# Установка переменных окружения
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data

# Открытие порта
EXPOSE 8080

# Запуск приложения через uvicorn вместо gunicorn
CMD ["uvicorn", \
     "src.web.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "4", \
     "--log-level", "info"]
