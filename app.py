# backend/app.py (versão final corrigida e robusta)
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
import os

app = Flask(__name__)
# Configuração do CORS para permitir requisições de qualquer origem durante o desenvolvimento
CORS(app)

# --- Variáveis de Ambiente ---
MONGO_URI = os.getenv('MONGO_URI', "mongodb+srv://gabriel:G8PESKXdYL2zWwrE@cluster0.cdtioo1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
MODEL_PATH = os.getenv('MODEL_PATH', 'model.pkl')

# --- Carregamento do Modelo de ML ---
try:
    with open(MODEL_PATH, "rb") as f:
        modelo_dict = pickle.load(f)
        pipeline_modelo = modelo_dict["pipeline"]
        label_encoder_modelo = modelo_dict["label_encoder"]
        print("✅ Modelo ML carregado com sucesso.")
except FileNotFoundError:
    print(f"⚠️ AVISO: Arquivo '{MODEL_PATH}' não encontrado. Rotas de ML podem não funcionar.")
    pipeline_modelo = None
    label_encoder_modelo = None

# --- Conexão com o MongoDB ---
try:
    client = MongoClient(MONGO_URI)
    db = client["test"]
    colecao_casos = db["cases"]
    colecao_vitimas = db["victims"]
    print("✅ Conexão com MongoDB estabelecida.")
except Exception as e:
    print(f"⚠️ ERRO ao conectar ao MongoDB: {e}")
    colecao_casos = None
    colecao_vitimas = None

# --- Middleware para verificar JSON ---
@app.after_request
def after_request(response):
    if request.path == '/':
        return response
    if not response.headers.get('Content-Type', '').startswith('application/json'):
        response.headers['Content-Type'] = 'application/json'
    return response

# --- Rotas Principais ---
@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "API de análise de casos criminais",
        "routes": {
            "test": "/api/test",
            "casos": "/api/casos",
            "vitimas": "/api/victims",
            "modelo": "/api/modelo/coeficientes"
        }
    })

@app.route('/api/test')
def test_route():
    return jsonify({"status": "ok", "message": "API Python está online!"})

# --- Funções Auxiliares ---
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

# --- Rotas de Dados ---
@app.route('/api/casos', methods=['GET'])
def listar_casos():
    if not colecao_casos:
        return jsonify({"error": "Conexão com MongoDB não disponível"}), 503
    
    try:
        casos_convertidos = [converter_documento_caso(doc) for doc in colecao_casos.find({})]
        return jsonify({
            "count": len(casos_convertidos),
            "data": casos_convertidos
        }), 200
    except Exception as e:
        return jsonify({"error": f"Erro ao listar casos: {str(e)}"}), 500

@app.route('/api/victims', methods=['GET'])
def listar_vitimas():
    if not colecao_vitimas:
        return jsonify({"error": "Conexão com MongoDB não disponível"}), 503
    
    try:
        projection = {
            "name": 1, "nic": 1, "gender": 1, 
            "age": 1, "identificationType": 1, 
            "ethnicity": 1, "cases": 1
        }
        vitimas_convertidas = [serialize_doc(v) for v in colecao_vitimas.find({}, projection)]
        return jsonify({
            "count": len(vitimas_convertidas),
            "data": vitimas_convertidas
        }), 200
    except Exception as e:
        return jsonify({"error": f"Erro ao listar vítimas: {str(e)}"}), 500

# --- Rotas de Machine Learning ---
@app.route('/api/modelo/coeficientes', methods=['GET'])
def coeficientes_modelo():
    if not pipeline_modelo:
        return jsonify({
            "error": "Modelo de Machine Learning não foi carregado",
            "solution": "Verifique se o arquivo model.pkl existe no servidor"
        }), 503
    
    try:
        # Extrai o classificador e o pré-processador do pipeline
        classifier = pipeline_modelo.named_steps['classifier']
        preprocessor = pipeline_modelo.named_steps['preprocessor']
        
        # Obtém os nomes das features categóricas após pré-processamento
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        
        # Obtém as importâncias das features
        importances = classifier.feature_importances_
        
        # Formata o resultado como dicionário
        features_importances = {
            feature: float(imp) 
            for feature, imp in zip(cat_features, importances)
        }
        
        return jsonify({
            "status": "success",
            "data": features_importances,
            "metadata": {
                "model_type": str(type(classifier).__name__,
                "features_count": len(features_importances)
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Não foi possível extrair coeficientes",
            "details": str(e)
        }), 500

# --- Health Check ---
@app.route('/api/health', methods=['GET'])
def health_check():
    status = {
        "python_api": "online",
        "mongo_connection": "online" if colecao_casos else "offline",
        "ml_model_loaded": "online" if pipeline_modelo else "offline"
    }
    return jsonify(status), 200

# --- Error Handlers ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint não encontrado",
        "message": "Verifique a URL e tente novamente"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Erro interno do servidor",
        "message": "Nossa equipe já foi notificada"
    }), 500

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    print(f"🚀 Iniciando servidor Flask na porta {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
