# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /backend

# Устанавливаем зависимости системы
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libpq-dev \
#     && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копируем проект
COPY . .

# Собираем статические файлы (если требуется)
# RUN python manage.py collectstatic --noinput

# Открываем порт
EXPOSE 8000

# Команда запуска
# CMD ["gunicorn", "tax_clustering.wsgi:application", "--bind", "0.0.0.0:8000"]
CMD ["sh", "-c", "python manage.py migrate && python manage.py shell < utils/create_superuser.py && python manage.py runserver 0.0.0.0:8000"]
