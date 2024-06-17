# Используем официальный образ Python
FROM ubuntu:20.04

# Установка рабочей директории в контейнере
ARG APP_DIR=/app
WORKDIR "$APP_DIR"

ENV PYTHONPATH "${PYTHONPATH}:pwd"

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get install python3-pip -y && \
    pip install --upgrade pip

# Копирование файлов зависимостей Python в контейнер
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего остального содержимого проекта в контейнер
COPY . .

# Открытие порта 7860 для Gradio
EXPOSE 7860

# Команда для запуска приложения
CMD ["python", "app.py"]