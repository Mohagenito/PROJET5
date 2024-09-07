# Projet5_Cat-gorisez automatiquement des questions: Classification Multi-Étiquette avec SBERT et Classifier Chain

## Objectif du Projet

Ce projet vise à développer un pipeline de classification multi-étiquette utilisant Sentence-BERT (SBERT) pour l'extraction de caractéristiques et une chaîne de classifieurs (`ClassifierChain`) pour la classification multi-étiquette. Il comprend à la fois des approches supervisées et non supervisées pour la classification des textes.

## Structure du Répertoire

Voici un aperçu de la structure des fichiers du projet :

- `.github/` : Dossier de configuration de GitHub, utilisé pour les actions GitHub, les workflows CI/CD, etc.
- `.gitattributes` : Fichier de configuration pour les attributs Git.
- `.gitignore` : Fichier spécifiant les fichiers et dossiers à ignorer par Git.
- `OUEDRAOGO_Mahamady_1_notebook_exploration_062024.ipynb` : Notebook Jupyter pour l'exploration et la préparation des données.
- `OUEDRAOGO_Mahamady_2_notebook_requete_API_062024.ipynb` : Notebook Jupyter pour l'exploration des données via des requêtes API.
- `OUEDRAOGO_Mahamady_3_notebook_approche_non_supervisée_062024.ipynb` : Notebook Jupyter utilisant une approche non supervisée pour la classification de textes.
- `OUEDRAOGO_Mahamady_4_notebook_approche_supervisée_062024.ipynb` : Notebook Jupyter utilisant une approche supervisée pour la classification de textes.
- `app.py` : Script Python pour l'API Flask permettant de faire des prédictions en production.
- `chain_classifier.pkl` : Modèle de classification multi-étiquette entraîné avec une chaîne de classifieurs.
- `mlb.pkl` : Modèle MultiLabelBinarizer utilisé pour la transformation des étiquettes.
- `sbert_model.pkl` : Modèle SBERT entraîné pour l'extraction de caractéristiques de texte.
- `requirements.txt` : Fichier listant les dépendances Python nécessaires pour exécuter le projet.
- `test_embedding.py` : Script de test pour valider le comportement des embeddings de texte.

## Prérequis

Assurez-vous d'avoir installé Python 3.8 ou une version ultérieure.
Vous pouvez utiliser `pip` pour installer les dépendances requises.

## Installation

1. Clonez le dépôt sur votre machine locale :

    ```bash
    git clone https://github.com/votre-utilisateur/votre-projet.git
    cd votre-projet
    ```

2. Installez les dépendances Python à l'aide de `pip` :

    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### 1. API de Prédiction

Le projet comprend une API Flask (`app.py`) qui permet de faire des prédictions sur de nouveaux textes. Pour exécuter l'API en mode développement :

```bash
python app.py
