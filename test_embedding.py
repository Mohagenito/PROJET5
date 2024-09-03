import unittest
import joblib  # Utiliser joblib pour charger le modèle
from sentence_transformers import SentenceTransformer

# Charger le modèle SBERT avec joblib
try:
    sbert_model = joblib.load('sbert_model.pkl')
except FileNotFoundError:
    print("Le fichier 'sbert_model.pkl' est introuvable. Vérifiez le chemin d'accès et réessayez.")

class TestEmbeddingFunction(unittest.TestCase):
    
    def test_embedding(self):
        """
        Test embedding of a sentence
        """
        try:
            question = "A regular string"
            embeddings = sbert_model.encode([question])
        except Exception as e:
            # If any exception is raised, the test should fail
            self.fail(f"sbert_model.encode([question]) raised an exception: {e}")
    
    def test_embedding_sentences(self):
        """
        Test embedding of several sentences
        """
        try:
            questions = ["A regular string", "A second string"]
            embeddings = sbert_model.encode(questions)
        except Exception as e:
            # If any exception is raised, the test should fail
            self.fail(f"sbert_model.encode(questions) raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
