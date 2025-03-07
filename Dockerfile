# Utiliser une image Python avec support GPU CUDA
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Installer Python et pip
RUN apt update && apt install -y python3 python3-pip git

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers dans le conteneur
COPY requirements.txt .
COPY main.py .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port pour FastAPI
EXPOSE 8000

# Commande pour exécuter l'application
CMD ["python3", "main.py"]