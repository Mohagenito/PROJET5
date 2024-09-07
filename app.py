from flask import Flask, request, jsonify
import joblib
import numpy as np

# Charger les modèles
try:
    chain_classifier = joblib.load('chain_classifier.pkl')
    sbert_model = joblib.load('sbert_model.pkl')
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
       
      
        
        # Encoder les phrases
        embeddings = sbert_model.encode([question])
        
        # Prédictions
        predictions = chain_classifier.predict(embeddings)
        
        # Inverse transform des prédictions pour obtenir les étiquettes d'origine
        labels = mlb.inverse_transform(predictions)
        
        return jsonify({'predictions': [list(label) for label in labels]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
