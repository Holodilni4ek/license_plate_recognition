# Базовый образ
FROM python:3.11-slim
# Автор Dockerfile
LABEL maintainer = "cadetstepan13@gmail.com"

# Установите Git и другие необходимые зависимости
RUN apt-get update && apt-get install -y git

WORKDIR /app
RUN git clone https://github.com/Holodilni4ek/RecognitionVLT.git

CMD ["python", "main.py"]