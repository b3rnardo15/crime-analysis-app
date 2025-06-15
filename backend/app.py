# backend/app.py (versão inicial)
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from pymongo import MongoClient
from dataclasses import dataclass, asdict
import random
from datetime import datetime, timedelta
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# Carregando o modelo treinado
with open("model.pkl", "rb") as f:
    modelo_dict = pickle.load(f)
    modelo = modelo_dict["pipeline"]
    label_encoder = modelo_dict["label_encoder"]

@app.route('/')
def hello():
    return "Bem-vindo à API de análise de casos criminais"

# MongoDB Connection
MONGO_URI = "mongodb+srv://gabriel:G8PESKXdYL2zWwrE@cluster0.cdtioo1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["test"]  
colecao = db["cases"] 

@dataclass
class Vitima:
    etnia: str
    idade: int

@dataclass
class Caso:
    data_do_caso: str
    tipo_do_caso: str
    localizacao: str
    vitima: Vitima

    def to_dict(self):
        return {
            "data_do_caso": self.data_do_caso,
            "tipo_do_caso": self.tipo_do_caso,
            "localizacao": self.localizacao,
            "vitima": asdict(self.vitima)
        }

# Função para converter documentos do Perioscan para o formato do dashboard
def converter_documento_perioscan(doc):
    # Extrair data do caso
    data_caso = doc.get("occurrenceDate", doc.get("openDate", doc.get("createdAt", "")))
    if isinstance(data_caso, datetime):
        data_caso = data_caso.isoformat()

    # Buscar informações da vítima relacionada
    vitima_info = None
    if "cases" in doc:
        for caso in doc["cases"]:
            if caso.get("relationType") == "principal":
                vitima_info = {
                    "identificationType": doc.get("identificationType", "não especificado"),
                    "gender": doc.get("gender", "não especificado"),
                    "nationality": doc.get("nationality", "não especificado"),
                    "bodyCondition": doc.get("bodyCondition", "não especificado"),
                    "ethnicity": doc.get("ethnicity", "não especificado")
                }
                break
    
    return {
        "data_do_caso": data_caso,
        "tipo_do_caso": doc.get("type", "não especificado"),
        "localizacao": doc.get("location", "Não especificado"),
        "status": doc.get("status", "Não especificado"),
        "titulo": doc.get("title", ""),
        "vitima": vitima_info
    }

def gerar_dados_aleatorios(n=20):
    tipos_casos = ["Furto", "Assalto", "Violencia domestica", "Trafico"]
    locais = ["Centro", "Bairro A", "Bairro B", "Zona Rural"]
    etnias = ["Branca", "Preta", "Parda", "Indigena", "Amarela"]
    casos = []
    base_date = datetime.now()
    
    for i in range(n):
        data_caso = (base_date - timedelta(days=random.randint(0, 365))).date().isoformat()
        caso = Caso(
            data_do_caso=data_caso,
            tipo_do_caso=random.choice(tipos_casos),
            localizacao=random.choice(locais),
            vitima=Vitima(
                etnia=random.choice(etnias),
                idade=random.randint(18, 90)
            )
        )
        casos.append(caso.to_dict())
    return casos

def validar_caso_json(data):
    try:
        vitima = data["vitima"]
        assert isinstance(vitima, dict)
        assert all(k in vitima for k in ("etnia", "idade"))
        assert isinstance(data["tipo_do_caso"], str)
        assert isinstance(data["localizacao"], str)
        return True
    except:
        return False

@app.route('/api/casos', methods=['GET'])
def listar_casos():
    try:
        # Buscar documentos da coleção cases do Perioscan
        documentos_perioscan = list(colecao.find({}))
        print(f"Encontrados {len(documentos_perioscan)} documentos no MongoDB")
        
        # Converter para o formato esperado pelo dashboard
        documentos_convertidos = []
        for doc in documentos_perioscan:
            # Remover o campo _id que não é serializável para JSON
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            
            # Converter para o formato do dashboard
            doc_convertido = converter_documento_perioscan(doc)
            documentos_convertidos.append(doc_convertido)
        
        print(f"Convertidos {len(documentos_convertidos)} documentos")
        print("Exemplo do primeiro documento convertido:", documentos_convertidos[0] if documentos_convertidos else "Nenhum documento")
        
        return jsonify(documentos_convertidos), 200
    except Exception as e:
        print(f"Erro ao listar casos: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/casos', methods=['POST'])
def criar_caso():
    data = request.get_json()
    if not data or not validar_caso_json(data):
        abort(400, "JSON inválido ou campos faltando.")
    colecao.insert_one(data)
    return jsonify({"message": "Caso criado com sucesso"}), 201

@app.route('/api/casos/<string:data_caso>', methods=['GET'])
def buscar_caso(data_caso):
    caso = colecao.find_one({"data_do_caso": data_caso}, {"_id": 0})
    if not caso:
        abort(404, "Caso não encontrado.")
    return jsonify(caso), 200

@app.route('/api/casos/<string:data_caso>', methods=['DELETE'])
def deletar_caso(data_caso):
    resultado = colecao.delete_one({"data_do_caso": data_caso})
    if resultado.deleted_count == 0:
        abort(404, "Caso não encontrado")
    return jsonify({"message": "Caso deletado"}), 200

@app.route('/api/modelo/coeficientes', methods=['GET'])
def coeficientes_modelo():
    try:
        colecao_vitimas = db["victims"]
        vitimas = list(colecao_vitimas.find({}))
        if len(vitimas) < 10:
            features_importances = {
                "idade": 0.35,
                "genero_masculino": 0.15,
                "genero_feminino": 0.15,
                "identificationType_identificada": 0.20,
                "nacionalidade_brasileira": 0.10,
                "bodyCondition_preservado": 0.05
            }
            return jsonify(features_importances), 200
        try:
            preprocessor = modelo.named_steps['preprocessor']
            classifier = modelo.named_steps['classifier']
            cat_encoder = preprocessor.named_transformers_['cat']
            cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[0][2])
            all_features = list(cat_features)
            importancias = classifier.feature_importances_
            features_importances = {
                feature: float(importance)
                for feature, importance in zip(all_features, importancias)
            }
            return jsonify(features_importances), 200
        except Exception:
            features_importances = {
                "idade": 0.35,
                "genero_masculino": 0.15,
                "genero_feminino": 0.15,
                "identificationType_identificada": 0.20,
                "nacionalidade_brasileira": 0.10,
                "bodyCondition_preservado": 0.05
            }
            return jsonify(features_importances), 200
    except Exception as e:
        print(f"Erro ao gerar coeficientes: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Pegando o pré-processador e o classificador XGBoost do pipeline
preprocessor = modelo.named_steps['preprocessor']
classifier = modelo.named_steps['classifier']

# Pegando nomes das features após o OneHotEncoding
cat_encoder = preprocessor.named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[0][2])
all_features = list(cat_features)

# Pegando as importâncias de feature do XGBoost
importancias = classifier.feature_importances_

features_importances = {
    feature: float(importance)
    for feature, importance in zip(all_features, importancias)
}
# Novas rotas para os algoritmos de ML e gráficos adicionais

@app.route('/api/casos/estatisticas/temporal', methods=['GET'])
def estatisticas_temporais():
    try:
        documentos = list(colecao.find({}))
        documentos_convertidos = [converter_documento_perioscan(doc) for doc in documentos]
        df = pd.DataFrame(documentos_convertidos)
        
        # Converter datas para datetime
        df['data_do_caso'] = pd.to_datetime(df['data_do_caso'], errors='coerce')
        
        # Filtrar registros com data válida
        df = df.dropna(subset=['data_do_caso'])
        
        # Agrupar por mês
        df['mes'] = df['data_do_caso'].dt.strftime('%Y-%m')
        contagem_mensal = df.groupby(['mes', 'tipo_do_caso']).size().reset_index(name='contagem')
        
        # Converter para formato adequado para o frontend
        resultado = contagem_mensal.to_dict('records')
        return jsonify(resultado), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/casos/estatisticas/boxplot', methods=['GET'])
def estatisticas_boxplot():
    try:
        # Usar a coleção victims ao invés de cases
        colecao_vitimas = db["victims"]
        vitimas = list(colecao_vitimas.find({}))
        
        # Converter para DataFrame
        df = pd.DataFrame(vitimas)
        
        # Verificar se há dados suficientes
        if len(df) == 0 or 'birthDate' not in df.columns:
            # Retornar dados simulados se não houver dados reais
            return jsonify({
                "identificada": {
                    "min": 18.0,
                    "q1": 25.0,
                    "median": 35.0,
                    "q3": 45.0,
                    "max": 65.0,
                    "outliers": []
                },
                "não identificada": {
                    "min": 20.0,
                    "q1": 30.0,
                    "median": 40.0,
                    "q3": 50.0,
                    "max": 70.0,
                    "outliers": []
                }
            }), 200
        
        # Calcular idades a partir da data de nascimento
        df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')
        df['idade'] = df['birthDate'].apply(lambda x: (datetime.now() - x).days / 365 if pd.notna(x) else None)
        
        # Usar identificationType como tipo
        resultado = {}
        for tipo in df['identificationType'].dropna().unique():
            idades = df[df['identificationType'] == tipo]['idade'].dropna()
            if len(idades) > 0:
                resultado[tipo] = {
                    'min': float(idades.min()),
                    'q1': float(idades.quantile(0.25)),
                    'median': float(idades.median()),
                    'q3': float(idades.quantile(0.75)),
                    'max': float(idades.max()),
                    'outliers': []
                }
        
        # Se não houver resultados, adicionar dados simulados
        if not resultado:
            resultado = {
                "identificada": {
                    "min": 18.0,
                    "q1": 25.0,
                    "median": 35.0,
                    "q3": 45.0,
                    "max": 65.0,
                    "outliers": []
                }
            }
        
        return jsonify(resultado), 200
    except Exception as e:
        print(f"Erro ao gerar estatísticas boxplot: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/clustering', methods=['GET'])
def clustering():
    try:
        # Usar a coleção victims ao invés de cases
        colecao_vitimas = db["victims"]
        vitimas = list(colecao_vitimas.find({}))
        
        # Converter para DataFrame
        df = pd.DataFrame(vitimas)
        
        # Verificar se há dados suficientes
        if len(df) < 3 or 'birthDate' not in df.columns or 'locationDate' not in df.columns:
            # Dados simulados para demonstração
            return jsonify([
                {
                    'cluster_id': 0,
                    'tamanho': 2,
                    'idade_media': 25.5,
                    'tipos_caso': {'identificada': 2},
                    'pontos': [
                        {'id': 0, 'x': 25.0, 'y': 1622505600.0, 'tipo': 'identificada'},
                        {'id': 1, 'x': 26.0, 'y': 1625097600.0, 'tipo': 'identificada'}
                    ]
                },
                {
                    'cluster_id': 1,
                    'tamanho': 2,
                    'idade_media': 35.0,
                    'tipos_caso': {'identificada': 2},
                    'pontos': [
                        {'id': 2, 'x': 34.0, 'y': 1627776000.0, 'tipo': 'identificada'},
                        {'id': 3, 'x': 36.0, 'y': 1630454400.0, 'tipo': 'identificada'}
                    ]
                }
            ]), 200
        
        # Calcular idades a partir da data de nascimento
        df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')
        df['idade'] = df['birthDate'].apply(lambda x: (datetime.now() - x).days / 365 if pd.notna(x) else None)
        
        # Usar locationDate como data do caso
        df['locationDate'] = pd.to_datetime(df['locationDate'], errors='coerce')
        # Converter para timestamp (número de segundos desde epoch)
        df['data_numerica'] = df['locationDate'].apply(lambda x: x.timestamp() if pd.notna(x) else None)
        
        # Selecionar apenas registros com dados completos
        df_clean = df.dropna(subset=['idade', 'data_numerica'])
        
        if len(df_clean) < 3:  # Precisamos de pelo menos 3 pontos para clustering
            # Dados simulados para demonstração
            return jsonify([
                {
                    'cluster_id': 0,
                    'tamanho': 2,
                    'idade_media': 25.5,
                    'tipos_caso': {'identificada': 2},
                    'pontos': [
                        {'id': 0, 'x': 25.0, 'y': 1622505600.0, 'tipo': 'identificada'},
                        {'id': 1, 'x': 26.0, 'y': 1625097600.0, 'tipo': 'identificada'}
                    ]
                },
                {
                    'cluster_id': 1,
                    'tamanho': 2,
                    'idade_media': 35.0,
                    'tipos_caso': {'identificada': 2},
                    'pontos': [
                        {'id': 2, 'x': 34.0, 'y': 1627776000.0, 'tipo': 'identificada'},
                        {'id': 3, 'x': 36.0, 'y': 1630454400.0, 'tipo': 'identificada'}
                    ]
                }
            ]), 200
        
        # Preparar dados para clustering
        X = df_clean[['idade', 'data_numerica']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determinar número ideal de clusters (simplificado)
        n_clusters = min(3, len(X_scaled))
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_clean['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Preparar resultado
        resultado = []
        for cluster_id in range(n_clusters):
            cluster_data = df_clean[df_clean['cluster'] == cluster_id]
            tipos_caso = {}
            
            # Verificar se a coluna identificationType existe
            if 'identificationType' in cluster_data.columns:
                for tipo, count in cluster_data['identificationType'].value_counts().items():
                    tipos_caso[tipo] = int(count)
            
            pontos = []
            for i, row in cluster_data.iterrows():
                pontos.append({
                    'id': int(i), 
                    'x': float(row['idade']), 
                    'y': float(row['data_numerica']), 
                    'tipo': row.get('identificationType', 'desconhecido')
                })
            
            resultado.append({
                'cluster_id': int(cluster_id),
                'tamanho': int(len(cluster_data)),
                'idade_media': float(cluster_data['idade'].mean()),
                'tipos_caso': tipos_caso,
                'pontos': pontos
            })
        
        return jsonify(resultado), 200
    except Exception as e:
        print(f"Erro ao gerar dados de clustering: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/regressao', methods=['GET'])
def regressao():
    try:
        # Usar a coleção victims ao invés de cases
        colecao_vitimas = db["victims"]
        vitimas = list(colecao_vitimas.find({}))
        
        # Verificar se há dados suficientes
        if len(vitimas) < 5 or not any('birthDate' in v for v in vitimas):
            # Dados simulados para demonstração
            return jsonify({
                'actual': [20, 25, 30, 35, 40],
                'predicted': [22, 26, 29, 36, 38],
                'r2': 0.85,
                'coeficientes': {
                    'identificationType_identificada': 2.5,
                    'gender_masculino': 1.8,
                    'nationality_brasileira': 0.9,
                    'mes': 0.3,
                    'intercept': 25.0
                }
            }), 200
        
        # Converter para DataFrame
        df = pd.DataFrame(vitimas)
        
        # Calcular idades a partir da data de nascimento
        df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')
        df['idade'] = df['birthDate'].apply(lambda x: (datetime.now() - x).days / 365 if pd.notna(x) else None)
        
        # Usar locationDate como data do caso
        df['locationDate'] = pd.to_datetime(df['locationDate'], errors='coerce')
        df['mes'] = df['locationDate'].dt.month.fillna(1).astype(int)
        
        # Verificar colunas categóricas disponíveis
        cat_columns = []
        for col in ['identificationType', 'gender', 'nationality']:
            if col in df.columns:
                cat_columns.append(col)
        
        # One-hot encoding para variáveis categóricas disponíveis
        if cat_columns:
            df_encoded = pd.get_dummies(df, columns=cat_columns)
        else:
            df_encoded = df.copy()
            # Adicionar colunas dummy simuladas se não houver categóricas
            df_encoded['dummy_feature'] = 1
        
        # Selecionar apenas registros com dados completos
        df_clean = df_encoded.dropna(subset=['idade'])
        
        if len(df_clean) < 5:  # Precisamos de dados suficientes para regressão
            # Dados simulados para demonstração
            return jsonify({
                'actual': [20, 25, 30, 35, 40],
                'predicted': [22, 26, 29, 36, 38],
                'r2': 0.85,
                'coeficientes': {
                    'identificationType_identificada': 2.5,
                    'gender_masculino': 1.8,
                    'nationality_brasileira': 0.9,
                    'mes': 0.3,
                    'intercept': 25.0
                }
            }), 200
        
        # Selecionar features e target
        feature_cols = [col for col in df_clean.columns if col.startswith('identificationType_') or 
                                                      col.startswith('gender_') or 
                                                      col.startswith('nationality_') or 
                                                      col == 'mes' or
                                                      col == 'dummy_feature']
        
        # Se não houver colunas suficientes, usar dados simulados
        if len(feature_cols) < 2:
            # Dados simulados para demonstração
            return jsonify({
                'actual': [20, 25, 30, 35, 40],
                'predicted': [22, 26, 29, 36, 38],
                'r2': 0.85,
                'coeficientes': {
                    'dummy_feature': 2.5,
                    'mes': 0.3,
                    'intercept': 25.0
                }
            }), 200
        
        # Treinar modelo de regressão com dados reais
        X = df_clean[feature_cols]
        y = df_clean['idade']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calcular previsões e R²
        y_pred = model.predict(X)
        r2 = model.score(X, y)
        
        # Preparar coeficientes
        coefs = {feature: float(coef) for feature, coef in zip(feature_cols, model.coef_)}
        coefs['intercept'] = float(model.intercept_)
        
        # Preparar dados para visualização
        visualization_data = {
            'actual': [float(val) for val in y.values],
            'predicted': [float(val) for val in y_pred],
            'r2': float(r2),
            'coeficientes': coefs
        }
        
        return jsonify(visualization_data), 200
    except Exception as e:
        print(f"Erro ao gerar dados de regressão: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/casos')
def get_casos():
    try:
        # Buscar todos os documentos da coleção
        documentos = list(colecao.find({}))
        print(f"Número de documentos encontrados: {len(documentos)}")
        
        # Converter documentos
        casos = [converter_documento_perioscan(doc) for doc in documentos]
        print(f"Primeiro caso convertido: {casos[0] if casos else 'Nenhum caso'}")
        
        return jsonify(casos)
    except Exception as e:
        print(f"Erro ao buscar casos: {str(e)}")
        return jsonify({"erro": str(e)}), 500

@app.route('/api/victims', methods=['GET'])
def listar_vitimas():
    try:
        # Usar a coleção victims ao invés de cases
        colecao_vitimas = db["victims"]
        
        # Buscar todas as vítimas
        vitimas = list(colecao_vitimas.find({}))
        
        # Função auxiliar para converter ObjectId para string
        def converter_objectid(valor):
            from bson import ObjectId
            if isinstance(valor, ObjectId):
                return str(valor)
            elif isinstance(valor, list):
                return [converter_objectid(item) for item in valor]
            elif isinstance(valor, dict):
                return {k: converter_objectid(v) for k, v in valor.items()}
            return valor

        # Converter todos os documentos
        vitimas_convertidas = []
        for vitima in vitimas:
            # Converter todos os campos recursivamente
            vitima_convertida = converter_objectid(vitima)
            
            # Converter datas para string ISO
            for campo_data in ['birthDate', 'locationDate', 'createdAt', 'updatedAt']:
                if campo_data in vitima_convertida and vitima_convertida[campo_data]:
                    vitima_convertida[campo_data] = vitima_convertida[campo_data].isoformat()
            
            vitimas_convertidas.append(vitima_convertida)
        
        print(f"Encontradas {len(vitimas_convertidas)} vítimas no MongoDB")
        return jsonify(vitimas_convertidas), 200
    
    except Exception as e:
        print(f"Erro ao listar vítimas: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if colecao.count_documents({}) == 0:
        print("Inserindo dados iniciais...")
        dados_iniciais = gerar_dados_aleatorios(20)
        colecao.insert_many(dados_iniciais)
    
    app.run(debug=True)