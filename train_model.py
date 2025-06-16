# train_model.py
# Este script treina um modelo de Machine Learning para prever o 'tipo de caso'
# com base em dados combinados das coleções 'cases' e 'victims'.

import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pickle

# --- 1. Conexão com o MongoDB ---
# Conecta ao banco de dados e obtém acesso às coleções.
MONGO_URI = "mongodb+srv://gabriel:G8PESKXdYL2zWwrE@cluster0.cdtioo1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["test"]
colecao_casos = db["cases"]
colecao_vitimas = db["victims"]

print("Conectado ao MongoDB com sucesso.")

# --- 2. Coleta e Combinação de Dados ---
# O objetivo é criar um dataset onde cada linha representa uma vítima em um caso específico.

# Primeiro, carregamos todos os casos e os colocamos em um dicionário para acesso rápido.
# A chave será o ID do caso (como string) e o valor será o documento do caso.
casos_dict = {str(case['_id']): case for case in colecao_casos.find({})}
print(f"Encontrados {len(casos_dict)} documentos na coleção 'cases'.")

# Agora, carregamos todas as vítimas.
vitimas_db = list(colecao_vitimas.find({}))
print(f"Encontrados {len(vitimas_db)} documentos na coleção 'victims'.")

# Vamos criar uma lista de registros combinados.
dados_combinados = []
for vitima in vitimas_db:
    # O campo 'cases' em 'victims' é um array de casos associados.
    if 'cases' in vitima and vitima['cases']:
        for relacao_caso in vitima['cases']:
            case_id = str(relacao_caso.get('caseId'))
            # Verificamos se o caso associado à vítima existe em nosso dicionário de casos.
            if case_id in casos_dict:
                caso = casos_dict[case_id]
                
                # Criamos um registro único com dados da vítima e do caso.
                registro = {
                    # Features (variáveis explicativas)
                    'location': caso.get('location'),
                    'status': caso.get('status'),
                    'gender': vitima.get('gender'),
                    'ethnicity': vitima.get('ethnicity'),
                    'bodyCondition': vitima.get('bodyCondition'),
                    'identificationType': vitima.get('identificationType'),
                    # Target (variável alvo)
                    'case_type': caso.get('type')
                }
                dados_combinados.append(registro)

print(f"Total de {len(dados_combinados)} registros combinados (vítima-caso) criados.")

# Se não houver dados, o script não pode continuar.
if not dados_combinados:
    print("Nenhum dado combinado encontrado. Não é possível treinar o modelo. Verifique se as vítimas estão associadas aos casos corretamente.")
    exit()

# --- 3. Preparação do DataFrame ---
# Convertendo a lista de dicionários para um DataFrame do Pandas.
df = pd.DataFrame(dados_combinados)

# Lidando com dados ausentes: preenchemos valores nulos com uma string 'desconhecido'.
# Isso é importante para que o OneHotEncoder funcione corretamente.
for column in df.columns:
    df[column] = df[column].fillna('desconhecido')

print("\nVisualização do DataFrame combinado:")
print(df.head())
print(f"\nTipos de caso únicos encontrados para treinamento:")
print(df["case_type"].value_counts())

# --- 4. Definição das Variáveis Explicativas (X) e Alvo (y) ---
# Selecionamos as colunas que serão usadas para prever o tipo de caso.
features = ['location', 'status', 'gender', 'ethnicity', 'bodyCondition', 'identificationType']
target = 'case_type'

X = df[features]
y = df[target]

# O modelo de classificação precisa que a variável alvo (y) seja numérica.
# Usamos o LabelEncoder para converter as strings (ex: "Furto") em números (ex: 0, 1, 2...).
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- 5. Criação do Pipeline de Pré-processamento e Modelagem ---
# O Pipeline automatiza as etapas de transformação dos dados e treinamento do modelo.

# Definimos quais colunas são categóricas.
categorical_features = features

# O ColumnTransformer aplica transformações específicas a colunas.
# Aqui, usamos o OneHotEncoder para converter nossas features categóricas em um formato numérico.
preprocessor = ColumnTransformer(
    transformers=[
        # O 'handle_unknown='ignore'' evita erros se o modelo encontrar uma categoria
        # nos dados de produção que não viu durante o treinamento.
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Contamos o número de classes únicas na variável alvo para configurar o classificador.
num_classes = len(label_encoder.classes_)
if num_classes < 2:
    print(f"Erro: Apenas {num_classes} tipo de caso encontrado. O modelo precisa de pelo menos 2 para ser treinado.")
    exit()

# Criamos o pipeline final.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        objective='multi:softprob', # Objetivo para classificação de múltiplas classes.
        num_class=num_classes,      # Informa ao modelo quantas classes existem.
        use_label_encoder=False,    # Desativamos o codificador interno do XGBoost, pois já fizemos isso.
        eval_metric='mlogloss'      # Métrica para avaliação do modelo.
    ))
])

# --- 6. Treinamento do Modelo ---
# O método .fit() executa todas as etapas do pipeline: pré-processamento e treinamento.
print("\nIniciando o treinamento do modelo...")
pipeline.fit(X, y_encoded)
print("Treinamento concluído com sucesso!")

# --- 7. Salvando o Modelo Treinado ---
# Salvamos o pipeline inteiro (pré-processador + modelo) e o label_encoder em um único arquivo.
# Isso garante que usaremos exatamente as mesmas transformações ao fazer previsões futuras.
with open("model.pkl", "wb") as f:
    pickle.dump({
        "pipeline": pipeline,
        "label_encoder": label_encoder
    }, f)

print("\nModelo treinado e salvo com sucesso no arquivo 'model.pkl'.")
