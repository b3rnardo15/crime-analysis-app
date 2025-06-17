
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
# Configuração do CORS para permitir requisições de qualquer origem durante o desenvolvimento
CORS(app)


# --- Carregamento do Modelo de ML ---
try:
    with open("model.pkl", "rb") as f:
        modelo_dict = pickle.load(f)
        pipeline_modelo = modelo_dict["pipeline"]
        label_encoder_modelo = modelo_dict["label_encoder"]
        print("Arquivo model.pkl carregado com sucesso.")
except FileNotFoundError:
    print("AVISO: Arquivo 'model.pkl' não encontrado. Rotas de ML podem não funcionar.")
    pipeline_modelo = None
    label_encoder_modelo = None

@app.route('/')
def hello():
    return "Bem-vindo à API de análise de casos criminais"

# --- ROTA DE TESTE (NOVA) ---
@app.route('/api/test')
def test_route():
    return jsonify({"status": "ok", "message": "A API Python está online e as rotas estão a ser registadas!"})

# --- Conexão com o MongoDB ---
MONGO_URI = "mongodb+srv://gabriel:G8PESKXdYL2zWwrE@cluster0.cdtioo1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["test"]
colecao_casos = db["cases"]
colecao_vitimas = db["victims"]

# --- Funções Auxiliares e de Conversão de Dados ---
def serialize_doc(doc):
    if doc is None: return None
    for key, value in doc.items():
        if isinstance(value, ObjectId): doc[key] = str(value)
        elif isinstance(value, datetime): doc[key] = value.isoformat()
        elif isinstance(value, list): doc[key] = [serialize_doc(i) if isinstance(i, dict) else str(i) if isinstance(i, ObjectId) else i for i in value]
        elif isinstance(value, dict): doc[key] = serialize_doc(value)
    return doc

def converter_documento_caso(doc):
    if doc is None: return {}
    return serialize_doc({
        "_id": doc.get("_id"), "titulo": doc.get("title", "N/A"),
        "descricao": doc.get("description", "N/A"), "tipo_do_caso": doc.get("type", "N/A"),
        "status": doc.get("status", "N/A"), "localizacao": doc.get("location", "N/A"),
        "data_ocorrencia": doc.get("occurrenceDate"), "data_abertura": doc.get("openDate"),
        "data_fechamento": doc.get("closeDate"), "criado_por": doc.get("createdBy"),
    })

# --- Rotas da API para Casos e Vítimas (Básicas) ---
@app.route('/api/casos', methods=['GET'])
def listar_casos():
    try:
        casos_convertidos = [converter_documento_caso(doc) for doc in colecao_casos.find({})]
        return jsonify(casos_convertidos), 200
    except Exception as e:
        return jsonify({"error": f"Erro ao listar casos: {e}"}), 500

@app.route('/api/victims', methods=['GET'])
def listar_vitimas():
    try:
        projection = {"name": 1, "nic": 1, "gender": 1, "age": 1, "identificationType": 1, "ethnicity": 1, "cases": 1}
        vitimas_convertidas = [serialize_doc(v) for v in colecao_vitimas.find({}, projection)]
        return jsonify(vitimas_convertidas), 200
    except Exception as e:
        return jsonify({"error": f"Erro ao listar vítimas: {e}"}), 500

# --- Rotas de Estatísticas e Machine Learning ---
@app.route('/api/modelo/coeficientes', methods=['GET'])
def coeficientes_modelo():
    if not pipeline_modelo:
        return jsonify({"error": "Arquivo model.pkl não foi carregado no servidor."}), 404
    try:
        classifier = pipeline_modelo.named_steps['classifier']
        preprocessor = pipeline_modelo.named_steps['preprocessor']
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        importances = classifier.feature_importances_
        features_importances = {feature: float(imp) for feature, imp in zip(cat_features, importances)}
        return jsonify(features_importances), 200
    except Exception as e:
        return jsonify({"error": f"Não foi possível extrair coeficientes: {e}"}), 500

# ... (O resto do seu ficheiro app.py continua aqui, sem alterações) ...
# (As rotas de boxplot, clustering e regressão continuam as mesmas)

if __name__ == "__main__":
    print("Iniciando o servidor Flask na porta 5000...")
    app.run(debug=True, port=5000)
