# Usa una imagen base de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el código de la aplicación (app.py, templates/, etc.)
COPY . .

# Expone el puerto que Render usará
# Render proporciona la variable $PORT automáticamente
EXPOSE $PORT

# El comando para iniciar la aplicación
# Usa gunicorn para producción
# Escucha en 0.0.0.0 en el puerto $PORT que Render asigna
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "4", "--timeout", "120"]
