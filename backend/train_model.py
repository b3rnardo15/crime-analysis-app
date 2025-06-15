import pandas as pd
from pymongo import MongoClient
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from xgboost import XGBClassifier

# Função para converter documentos do Perioscan para o formato do dashboard
def converter_documento_perioscan(doc):
    return {
        "data_do_caso": doc.get("occurrenceDate", doc.get("openDate", doc.get("createdAt", ""))),
        "tipo_do_caso": doc.get("type", "nao especificado"),
        "localizacao": doc.get("location", "Não especificado"),
        "status": doc.get("status", "Não especificado"),
        "titulo": doc.get("title", "")
    }

# 1. Conectar no MongoDB e puxar dados
MONGO_URI = "mongodb+srv://gabriel:G8PESKXdYL2zWwrE@cluster0.cdtioo1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["test"]
colecao = db["cases"]

dados_perioscan = list(colecao.find({}))
print(f"Número total de casos encontrados: {len(dados_perioscan)}")

dados = [converter_documento_perioscan(doc) for doc in dados_perioscan]
print(f"Número de casos após conversão: {len(dados)}")

# 2. Preparar DataFrame flat
lista = []
for d in dados:
    lista.append({
        "tipo_do_caso": d["tipo_do_caso"],
        "localizacao": d["localizacao"],
        "status": d["status"],
        "data_do_caso": d["data_do_caso"]
    })

df = pd.DataFrame(lista)
print(f"\nTipos de casos únicos encontrados:")
print(df["tipo_do_caso"].value_counts())

# 3. Variáveis explicativas e alvo
x = df[["localizacao"]]
y = df["tipo_do_caso"]

# 4. Encode da variável alvo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5. Pipeline 
categorical_features = ["localizacao"] 
numeric_features = []

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Antes de criar o pipeline, vamos contar o número de classes únicas
num_classes = len(label_encoder.classes_)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(objective='multi:softprob', num_class=num_classes))
])

# 6. Treinar
pipeline.fit(x, y_encoded)

# 7. Salvar pipeline + label encoder
with open("model.pkl", "wb") as f:
    pickle.dump({
        "pipeline": pipeline,
        "label_encoder": label_encoder
    }, f)

print("Modelo treinado e salvo em model.pkl")
