# Базовый образ
FROM python:3.12-slim as python-base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Билдер
FROM python-base as builder
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    build-essential

# Установка зависимостей
WORKDIR $PYSETUP_PATH
COPY requirements.txt ./requirements.txt
RUN python -m venv $VENV_PATH \
    && . $VENV_PATH/bin/activate \
    && pip install --no-cache-dir -r requirements.txt

# Финальный образ
FROM python-base as production
COPY --from=builder $VENV_PATH $VENV_PATH

# Копирование кода
WORKDIR /app
COPY src/ ./src/

# Копирование файла учетных данных Firebase
COPY rag-project-6fbb6-firebase-adminsdk-oiud2-87842f7d56.json /app/rag-project-6fbb6-firebase-adminsdk-oiud2-87842f7d56.json
ENV FIREBASE_CREDENTIALS_PATH=/app/rag-project-6fbb6-firebase-adminsdk-oiud2-87842f7d56.json

# Запуск
CMD ["sh", "-c", "uvicorn src.web.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
