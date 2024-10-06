from flask import Flask, request, jsonify  # Importation des modules nécessaires de Flask pour créer l'application web.
import joblib  # Importation de joblib pour charger des modèles machine learning pré-entraînés.
import numpy as np  # Importation de NumPy, souvent utilisé pour manipuler des tableaux et des vecteurs.
import requests  # Pour télécharger les fichiers depuis Google Drive.
import os  # Pour vérifier l'existence des fichiers localement.

# Fonction pour télécharger un fichier à partir de Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

# Fonction pour obtenir un token de confirmation pour les fichiers volumineux
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

# Fonction pour sauvegarder le contenu téléchargé dans un fichier
def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Télécharger les modèles s'ils ne sont pas présents localement
if not os.path.exists('sbert_model.pkl'):
    print("Téléchargement du modèle SBERT...")
    download_file_from_google_drive('1dd3wJNko1iWnD37PwdNKVrfo409CtgMr', 'sbert_model.pkl')

if not os.path.exists('chain_classifier.pkl'):
    print("Téléchargement du Classifier Chain...")
    download_file_from_google_drive('1PJi0L2n4OBY1LvBpWfhDFjm7RkAMaFW1', 'chain_classifier.pkl')

if not os.path.exists('mlb.pkl'):
    print("Téléchargement du MultiLabel Binarizer...")
    download_file_from_google_drive('1LiIU7RcJycJUaW1k3tjLActsT_9o2W3j', 'mlb.pkl')

# Charger les modèles
try:
    # Charger le modèle Classifier Chain sauvegardé avec joblib
    chain_classifier = joblib.load('chain_classifier.pkl')
    
    # Charger le modèle SBERT (Sentence-BERT) pour l'encodage des phrases
    sbert_model = joblib.load('sbert_model.pkl')
    
    # Charger l'encodeur multi-label Binarizer pour obtenir les étiquettes originales
    mlb = joblib.load('mlb.pkl')
    
    print("Les modèles sont chargés correctement.")  # Confirmation que les modèles sont chargés avec succès.
except Exception as e:
    # Gestion des erreurs si le chargement des modèles échoue
    print(f"Erreur lors du chargement des modèles : {e}")

# Créer une instance de l'application Flask
app = Flask(__name__)

# Définir une route de base pour vérifier que l'API fonctionne
@app.route('/')
def home():
    return "API de prédiction avec SBERT et Classifier Chain est en cours d'exécution."

# Définir une route qui prend un paramètre `first_name` et renvoie un message de salutation
@app.route('/salut_perso/<string:first_name>')
def salut_toi(first_name):
    return f"Salut {first_name} !"  # Retourne une salutation personnalisée

# Définir une route de prédiction qui prend une question en tant que paramètre
@app.route('/predict/<string:question>')
def predict(question):
    try:
        # Encoder la question donnée en utilisant SBERT pour obtenir ses embeddings (représentation vectorielle)
        embeddings = sbert_model.encode([question])
        
        # Utiliser le modèle Classifier Chain pour prédire les étiquettes de la question encodée
        predictions = chain_classifier.predict(embeddings)
        
        # Transformer les prédictions pour retrouver les étiquettes d'origine (débinairisation)
        labels = mlb.inverse_transform(predictions)
        
        # Retourner les étiquettes sous forme de liste JSON
        return jsonify({'predictions': [list(label) for label in labels]})
    except Exception as e:
        # En cas d'erreur pendant la prédiction, retourner un message d'erreur JSON
        return jsonify({'error': str(e)})

# Point d'entrée de l'application
if __name__ == '__main__':
    # Lancer l'application Flask en mode debug, accessible depuis n'importe quelle adresse IP sur le port 5001
    app.run(debug=True, host='0.0.0.0', port=5001)
