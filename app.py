from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

# Charger les modèles
try:
    # Charger le modèle Classifier Chain sauvegardé avec joblib
    chain_classifier = joblib.load('chain_classifier.pkl')
    
    # Charger directement le modèle SBERT à partir de SentenceTransformer
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Utilise le nom du modèle SBERT que tu as utilisé
    
    # Charger l'encodeur multi-label Binarizer
    mlb = joblib.load('mlb.pkl')
    
    print("Les modèles sont chargés correctement.")
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {e}")

app = Flask(__name__)

@app.route('/')
def home():
    return "API de prédiction avec SBERT et Classifier Chain est en cours d'exécution."

@app.route('/salut_perso/<string:first_name>')
def salut_toi(first_name):
    return f"Salut {first_name} !"

@app.route('/predict/<string:question>')
def predict(question):
    try:
        # Encoder la question avec SBERT
        embeddings = sbert_model.encode([question])
        
        # Prédire les étiquettes avec Classifier Chain
        predictions = chain_classifier.predict(embeddings)
        
        # Transformer les prédictions en étiquettes originales
        labels = mlb.inverse_transform(predictions)
        
        return jsonify({'predictions': [list(label) for label in labels]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
