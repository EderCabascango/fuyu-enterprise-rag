# Usamos una imagen ligera de Python 3.12 (la versión estable que instalamos)
FROM python:3.12-slim

# Directorio de trabajo
WORKDIR /app

# Copiamos los archivos necesarios
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponemos el puerto que usa Chainlit (8000 por defecto)
EXPOSE 8000

# Comando para arrancar Chainlit en modo producción
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]