from flask import Flask, request, jsonify  # Importation des modules nécessaires de Flask pour créer l'application web.
import joblib  # Importation de joblib pour charger des modèles machine learning pré-entraînés.
import numpy as np  # Importation de NumPy, souvent utilisé pour manipuler des tableaux et des vecteurs.

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
