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
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('stopwords'); \
    nltk.download('wordnet'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('punkt_tab')"

# Установка переменных окружения
ENV FLASK_APP=src.web.app
ENV PYTHONPATH=/app

# Открытие порта
EXPOSE 8080

# Копируем wsgi файл
COPY src/web/wsgi.py .

# Меняем команду запуска
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--worker-class", "eventlet", "--timeout", "120", "wsgi:app"]
