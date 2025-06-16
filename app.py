# backend/app.py (versão final, corrigida e robusta)
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

@app.route('/api/casos/estatisticas/boxplot', methods=['GET'])
def estatisticas_boxplot():
    try:
        vitimas = list(colecao_vitimas.find({}, {"birthDate": 1, "identificationType": 1}))
        if not vitimas: return jsonify({"error": "Nenhuma vítima encontrada."}), 400
        df = pd.DataFrame(vitimas)
        if 'birthDate' not in df.columns or 'identificationType' not in df.columns:
            return jsonify({"error": "Campos 'birthDate' ou 'identificationType' ausentes."}), 400

        df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')
        df['idade'] = (datetime.now() - df['birthDate']).dt.days / 365.25
        df = df.dropna(subset=['idade', 'identificationType'])
        if df.empty:
            return jsonify({"error": "Nenhum dado de idade válido para gerar o gráfico."}), 400
        resultado = {}
        for tipo in df['identificationType'].unique():
            idades = df[df['identificationType'] == tipo]['idade']
            if not idades.empty:
                resultado[tipo] = {'min': float(idades.min()), 'q1': float(idades.quantile(0.25)), 'median': float(idades.median()), 'q3': float(idades.quantile(0.75)), 'max': float(idades.max()), 'outliers': []}
        return jsonify(resultado), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/clustering', methods=['GET'])
def clustering():
    try:
        vitimas = list(colecao_vitimas.find({}, {"_id": 1, "birthDate": 1, "locationDate": 1, "identificationType": 1}))
        if len(vitimas) < 3: return jsonify({"error": "Dados insuficientes para clustering (menos de 3 vítimas)."}), 400
        
        df = pd.DataFrame(vitimas)
        required_cols = ['birthDate', 'locationDate', 'identificationType']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             return jsonify({"error": f"Dados de vítimas incompletos. Faltam colunas: {', '.join(missing_cols)}"}), 400
        
        df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')
        df['idade'] = (datetime.now() - df['birthDate']).dt.days / 365.25
        df['locationDate'] = pd.to_datetime(df['locationDate'], errors='coerce')
        df['data_numerica'] = df['locationDate'].apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
        
        df_clean = df.dropna(subset=['idade', 'data_numerica', 'identificationType']).copy()
        
        if len(df_clean) < 3: return jsonify({"error": f"Dados válidos insuficientes após limpeza ({len(df_clean)} de 3 necessários)."}), 400

        X = df_clean[['idade', 'data_numerica']].values
        X_scaled = StandardScaler().fit_transform(X)
        n_clusters = min(3, len(df_clean))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        df_clean.loc[:, 'cluster'] = kmeans.fit_predict(X_scaled)
        
        resultado = []
        for cluster_id in range(n_clusters):
            cluster_data = df_clean[df_clean['cluster'] == cluster_id]
            pontos = [{'id': str(row['_id']), 'x': float(row['idade']), 'y': float(row['data_numerica']), 'tipo': row.get('identificationType', 'N/A')} for _, row in cluster_data.iterrows()]
            resultado.append({'cluster_id': int(cluster_id), 'tamanho': len(cluster_data), 'idade_media': float(cluster_data['idade'].mean()), 'tipos_caso': cluster_data['identificationType'].value_counts().to_dict(), 'pontos': pontos})
        return jsonify(resultado), 200
    except Exception as e:
        print(f"ERRO CRÍTICO NO CLUSTERING: {e}")
        return jsonify({"error": f"Erro interno do servidor no clustering: {e}"}), 500

@app.route('/api/ml/regressao', methods=['GET'])
def regressao():
    try:
        vitimas = list(colecao_vitimas.find({}, {"birthDate": 1, "locationDate": 1, "identificationType": 1, "gender": 1, "nationality": 1}))
        if len(vitimas) < 5: return jsonify({"error": "Dados insuficientes para regressão (menos de 5 vítimas)."}), 400

        df = pd.DataFrame(vitimas)
        
        required_cat_cols = ['identificationType', 'gender', 'nationality']
        available_cat_cols = [col for col in required_cat_cols if col in df.columns]
        if not available_cat_cols:
            return jsonify({"error": f"Nenhuma coluna categórica ({', '.join(required_cat_cols)}) encontrada para regressão."}), 400
        
        if 'birthDate' not in df.columns:
            return jsonify({"error": "Coluna 'birthDate' é essencial para a regressão e não foi encontrada."}), 400

        df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')
        df['idade'] = (datetime.now() - df['birthDate']).dt.days / 365.25
        df['locationDate'] = pd.to_datetime(df.get('locationDate'), errors='coerce')
        df['mes'] = df['locationDate'].dt.month.fillna(0)
        
        df_encoded = pd.get_dummies(df, columns=available_cat_cols, dummy_na=True)
        
        # *** CORREÇÃO CRÍTICA AQUI: Adicionado .copy() para evitar o SettingWithCopyWarning e o crash ***
        df_clean = df_encoded.dropna(subset=['idade']).copy()
        
        if len(df_clean) < 5: return jsonify({"error": f"Dados válidos insuficientes após limpeza ({len(df_clean)} de 5 necessários)."}), 400

        feature_cols = [col for col in df_clean.columns if any(col.startswith(p) for p in available_cat_cols) or col == 'mes']
        if not feature_cols: return jsonify({"error": "Nenhuma feature para o modelo após o encoding."}), 400

        X = df_clean[feature_cols]
        y = df_clean['idade']
        
        if X.empty or y.empty:
            return jsonify({"error": "Não foi possível criar o conjunto de dados de treino. Verifique os dados categóricos."}), 400

        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        coefs = {feature: float(coef) for feature, coef in zip(feature_cols, model.coef_)}
        coefs['intercept'] = float(model.intercept_)
        return jsonify({'actual': y.tolist(), 'predicted': y_pred.tolist(), 'r2': float(model.score(X, y)), 'coeficientes': coefs}), 200
    except Exception as e:
        print(f"ERRO CRÍTICO NA REGRESSÃO: {e}")
        return jsonify({"error": f"Erro interno do servidor na regressão: {e}"}), 500

if __name__ == "__main__":
    print("Iniciando o servidor Flask na porta 5000...")
    app.run(debug=True, port=5000)
