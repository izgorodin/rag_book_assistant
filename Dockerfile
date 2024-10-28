# Базовый образ
FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
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
RUN mkdir -p uploads logs

# Устанавливаем NLTK данные
# Загрузка NLTK данных
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('punkt_tab'); \
    nltk.download('stopwords'); \
    nltk.download('wordnet'); \
    nltk.download('words'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('averaged_perceptron_tagger_eng'); \
    nltk.download('omw-1.4'); \
    nltk.download('tagsets'); \
    nltk.download('maxent_ne_chunker_tab'); \
    nltk.download('universal_tagset')"

# Установка переменных окружения
ENV FLASK_APP=src.web.app
ENV PYTHONPATH=/app

# Открытие порта
EXPOSE 8080

# Копируем WSGI файл
COPY src/web/wsgi.py /app/src/web/wsgi.py

# Меняем команду запуска
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--worker-class", "eventlet", "--timeout", "120", "src.web.wsgi:app"]
