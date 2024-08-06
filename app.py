from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Charger le modèle depuis le fichier pickle
with open('./models/decision_tree_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


@app.route('/')
def hello():
    return "hello world"

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Récupérer les données JSON de la requête
#     data = request.get_json()

#     # Convertir les données en DataFrame
#     df = pd.DataFrame(data)

#     msisdns = df.pop('MSISDN').tolist()

#     # Assurez-vous que l'ordre des colonnes correspond à celui utilisé pour entraîner le modèle
#     # Faire des prédictions
#     predictions = loaded_model.predict(df)

#     # Retourner les prédictions en format JSON
#     # result = {'MSISDN': msisdns, 'predictions': predictions.tolist()}
    
#     return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer le fichier CSV de la requête
    file = request.files['file']
    
    # Lire le fichier CSV dans un DataFrame
    df = pd.read_csv(file)
    print(df.head())

    # msisdns = df.pop('MSISDN').tolist()
    msisdns = df["MSISDN"]

    # Assurez-vous que l'ordre des colonnes correspond à celui utilisé pour entraîner le modèle
    # Faire des prédictions
    predictions = loaded_model.predict(df)

    # Retourner les prédictions en format JSON
    result = dict(zip(msisdns, predictions.tolist()))
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
