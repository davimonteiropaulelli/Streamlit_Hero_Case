import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Carregando os modelos treinados
model_path = "C:\\Users\\dpaul\\OneDrive\\Documentos\\Heroes_Case_Alelo\\Models"
with open(f"{model_path}\\naive_bayes_model.pkl", 'rb') as file:
    naive_bayes_model = pickle.load(file)
with open(f"{model_path}\\mlp_model.pkl", 'rb') as file:
    mlp_model = pickle.load(file)
with open(f"{model_path}\\xgb_model.pkl", 'rb') as file:
    xgb_model = pickle.load(file)

# Carregando o conjunto de dados tratado
data_path = "C:\\Users\\dpaul\\OneDrive\\Documentos\\Heroes_Case_Alelo\\Dataset\\Treated\\treated_data.csv"
df = pd.read_csv(data_path)
df = df.drop(columns=["Unnamed: 0"])

# Convertendo altura e peso para float
df['Height'] = df['Height'].astype(float)
df['Weight'] = df['Weight'].astype(float)

# Criando um novo LabelEncoder para a coluna 'Gender'
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

# Mapeamento do alinhamento
alignment_mapping = {0: "ruim", 1: "bom"}  # Ajuste conforme necessário

# Configurando a aplicação Streamlit
st.title("Aplicação Interativa de Super-Heróis com Modelos de Machine Learning")

# **Exploração de Dados**
st.header("Exploração de Dados")
st.write("Explore os dados de super-heróis.")

# Mostrar as primeiras linhas do conjunto de dados
if st.checkbox("Mostrar informações de super-heróis"):
    st.write(df.head())

# Estatísticas descritivas
if st.checkbox("Mostrar estatísticas descritivas"):
    st.write(df.describe(percentiles=[.25, .50, .75, .80, .85, .90, .95, .97, .99, .999, .9999]))

# Filtros interativos
st.subheader("Filtrar super-heróis por critérios")
alignment = st.selectbox("Alinhamento", df['Alignment'].unique())
gender = st.selectbox("Gênero", le_gender.classes_)  # Usando as classes corretas do novo LabelEncoder
publisher = st.selectbox("Editora", df['Publisher'].unique())

# Aplicar filtros
filtered_heroes = df[
    (df['Alignment'] == alignment) & 
    (df['Gender'] == le_gender.transform([gender])[0]) & 
    (df['Publisher'] == publisher)
]
st.write("Super-heróis filtrados:", filtered_heroes)

# **Resultados do Clustering**
st.header("Resultados do Clustering")
st.write("Visualize os clusters de super-heróis.")

# Gráfico de dispersão melhorado para clustering
if st.checkbox("Mostrar gráfico de Dispersão de clusters"):
    # Simulando dados de clusterização para o exemplo
    x = df['Height']
    y = df['Weight']
    labels = np.random.randint(0, 3, len(df))

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(x, y, c=labels, cmap='viridis', alpha=0.6, edgecolors='w', s=100)
    plt.xlabel("Altura")
    plt.ylabel("Peso")
    plt.title("Clusters de Super-Heróis (Exemplo por Altura e Peso)")
    plt.colorbar(scatter, label="Cluster")
    st.pyplot(fig)

# Filtrando apenas as colunas numéricas para calcular a matriz de correlação
df_corr_matrix = df.select_dtypes(include=[float, int]).corr()

# Mapa de calor das correlações
if st.checkbox("Mostrar mapa de calor das correlações"):
    corr_matrix = df_corr_matrix.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.title("Mapa de Calor das Correlações")
    st.pyplot(fig)

# **Classificação do Alinhamento**
st.header("Classificação do Alinhamento")
st.write("Use o modelo Naive Bayes ou MLP para prever o alinhamento de um super-herói.")

# Entradas do usuário
height = st.number_input("Altura do Super-Herói", min_value=0.0, format="%.2f")
weight = st.number_input("Peso do Super-Herói", min_value=0.0, format="%.2f")
gender_input = st.selectbox("Gênero do Super-Herói", le_gender.classes_)
gender_encoded = le_gender.transform([gender_input])[0]

# Previsão usando Naive Bayes
if st.button("Prever Alinhamento com Naive Bayes"):
    prediction_nb = naive_bayes_model.predict([[height, weight, gender_encoded]])
    st.write(f"Previsão de Alinhamento (Naive Bayes): {alignment_mapping[prediction_nb[0]]}")

# Previsão usando MLP
if st.button("Prever Alinhamento com MLPClassifier"):
    prediction_mlp = mlp_model.predict([[height, weight, gender_encoded]])
    st.write(f"Previsão de Alinhamento (MLPClassifier): {alignment_mapping[prediction_mlp[0]]}")

# **Previsão de Peso**
st.header("Previsão de Peso")
st.write("Use o modelo XGBoost para prever o peso de um super-herói com base em suas características.")

# Entradas para previsão de peso
height_for_weight = st.number_input("Altura para Previsão de Peso", min_value=0.0, format="%.2f")
gender_for_weight = st.selectbox("Gênero para Previsão de Peso", le_gender.classes_)
gender_for_weight_encoded = le_gender.transform([gender_for_weight])[0]

# Previsão de peso usando XGBoost
if st.button("Prever Peso"):
    weight_prediction = xgb_model.predict([[height_for_weight, gender_for_weight_encoded]])
    st.write(f"Peso Previsto: {weight_prediction[0]:.2f} kg")

# Rodapé
st.write("\nDesenvolvido com ❤️ usando Streamlit. Espero que goste :)")
