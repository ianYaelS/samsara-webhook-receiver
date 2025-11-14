# Usar una imagen base de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requisitos primero para aprovechar el caché de Docker
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Render establece la variable de entorno $PORT automáticamente.
# Este comando usa "sh -c" para permitir que la variable $PORT se expanda.
# Uvicorn se ejecutará en el host 0.0.0.0 y en el puerto que Render asigne.
CMD sh -c "uvicorn webhook_receiver:app --host 0.0.0.0 --port ${PORT:-8000}"
