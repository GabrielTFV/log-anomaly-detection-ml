#!/usr/bin/env python
# coding: utf-8

# In[1]:


# --- Importações Essenciais ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# --- Importações do Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configurações de Visualização ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

print("Bibliotecas importadas e configurações aplicadas.")


# In[2]:


# Carregar o dataset a partir do diretório de dados brutos
try:
    # O separador deste CSV é vírgula seguida de espaço, por isso usamos o engine python e um regex.
    df = pd.read_csv('../../data/raw/cybersecurity_logs_cidds.csv', sep=r',\s*', engine='python')
    print("Dataset carregado com sucesso!")
    print(f"O dataset contém {df.shape[0]} linhas e {df.shape[1]} colunas.")
except FileNotFoundError:
    print("Erro: O arquivo 'cybersecurity_logs_cidds.csv' não foi encontrado.")
    print("Por favor, coloque o dataset no diretório 'data/raw/'.")

# Exibir as primeiras linhas e informações básicas para verificação
if 'df' in locals():
    print("\n--- Primeiras 5 linhas do dataset ---")
    print(df.head())
    print("\n--- Informações do DataFrame ---")
    df.info()


# In[3]:


if 'df' in locals():
    print("\n--- Análise da Variável Alvo ('class') ---")
    print("Distribuição original da coluna 'class':")
    print(df['class'].value_counts(normalize=True))

    # Criar a flag binária de anomalia
    # Consideramos 'normal' como 0 e todo o resto (suspicious, unknown) como 1 (anomalia)
    df['Anomaly_Flag'] = df['class'].apply(lambda x: 0 if x == 'normal' else 1)

    print("\nDistribuição da nova coluna 'Anomaly_Flag':")
    print(df['Anomaly_Flag'].value_counts(normalize=True))

    # Visualizar a distribuição
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Anomaly_Flag', data=df)
    plt.title('Distribuição de Logs Normais (0) vs. Anômalos (1)')
    plt.ylabel('Contagem')
    plt.show()

    # Remover colunas que não serão usadas na modelagem ou que são redundantes
    # attackID, attackDescription e attackType são explicações do label 'class'
    df.drop(columns=['class', 'attackID', 'attackDescription', 'attackType'], inplace=True)
    print("\nColunas redundantes removidas. DataFrame pronto para EDA e engenharia de features.")


# In[4]:


if 'df' in locals():
    print("\n--- Análise Exploratória de Dados (EDA) ---")

    # 1. Análise de Variáveis Numéricas
    numeric_cols_to_analyze = ['Duration', 'Packets', 'Bytes', 'Flows']
    print(f"Analisando distribuições para: {numeric_cols_to_analyze}")
    df[numeric_cols_to_analyze].describe()
    # Tratamento global de infinities ou NaNs
    df[numeric_cols_to_analyze] = df[numeric_cols_to_analyze].apply(pd.to_numeric, errors='coerce')
    for col in numeric_cols_to_analyze:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df = df[df[col] > 0]  # remove zeros, pois log_scale não aceita zero
        df = df.dropna(subset=[col])

    for col in numeric_cols_to_analyze:
        print(col, np.isnan(df[col]).sum(), np.isinf(df[col]).sum(), (df[col] == 0).sum())

    # Visualização (usando escala de log para melhor visualização devido a outliers)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for i, col in enumerate(numeric_cols_to_analyze):
        sns.kdeplot(data=df, x=col, hue='Anomaly_Flag', ax=axes[i//2, i%2], log_scale=True, warn_singular=False)
        axes[i//2, i%2].set_title(f'Distribuição de {col} (Escala Log)')
    plt.tight_layout()
    plt.show()

    # 2. Análise de Variáveis Categóricas
    categorical_cols_to_analyze = ['Proto', 'Flags']
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for i, col in enumerate(categorical_cols_to_analyze):
        sns.countplot(data=df, x=col, hue='Anomaly_Flag', ax=axes[i], order=df[col].value_counts().index[:10])
        axes[i].set_title(f'Distribuição de {col} (Top 10)')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


# In[5]:


if 'df' in locals():
    print("\n--- Engenharia de Features ---")

    # 1. Converter 'Date first seen' para Timestamp
    df['Timestamp'] = pd.to_datetime(df['Date first seen'])
    df.drop(columns=['Date first seen'], inplace=True)

    # 2. Criar o identificador de fluxo (flow_id)
    df['flow_id'] = df['Src IP Addr'].astype(str) + '-' + df['Dst IP Addr'].astype(str)

    # 3. Ordenar por fluxo e depois por tempo (ESSENCIAL para cálculos corretos)
    df = df.sort_values(by=['flow_id', 'Timestamp']).reset_index(drop=True)
    print("DataFrame ordenado por 'flow_id' e 'Timestamp'.")

    # 4. Criar features temporais básicas
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    print("Features temporais ('hour', 'day_of_week') criadas.")

    # 5. Criar features de interação e de fluxo
    # O cálculo agora será correto devido à ordenação
    df['Bytes_per_Packet'] = df['Bytes'] / (df['Packets'] + 1e-6)
    df['Packets_per_Second'] = df['Packets'] / (df['Duration'] + 1e-6)
    df['time_diff'] = df.groupby('flow_id')['Timestamp'].diff().dt.total_seconds().fillna(0)
    df['event_in_flow_count'] = df.groupby('flow_id').cumcount() + 1
    print("Features de interação e de fluxo criadas.")

    print("\n--- DataFrame após engenharia de features ---")
    print(df[['Timestamp', 'flow_id', 'time_diff', 'event_in_flow_count', 'Anomaly_Flag']].head())


# In[6]:


if 'df' in locals():
    print("\n--- Definição do Pipeline de Pré-processamento ---")

    # 1. Colunas por tipo
    numeric_features = [
        'Duration', 'Src Pt', 'Dst Pt', 'Packets', 'Bytes', 'Flows', 'Tos',
        'hour', 'day_of_week', 'Bytes_per_Packet', 'Packets_per_Second',
        'time_diff', 'event_in_flow_count'
    ]
    categorical_features_low = ['Proto', 'Flags']
    # Usaremos FeatureHasher para alta cardinalidade para controlar a dimensionalidade
    categorical_features_high = ['Src IP Addr', 'Dst IP Addr']

    # 2. Criar os pipelines de transformação
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer_low = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    def to_dicts(X_df):
        return X_df.to_dict(orient='records')

    categorical_transformer_high = Pipeline(steps=[
        ('to_dicts', FunctionTransformer(to_dicts, validate=False)),
        ('hasher', FeatureHasher(n_features=256, input_type='dict'))
    ])

    # 3. Combinar com ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat_low', categorical_transformer_low, categorical_features_low),
            ('cat_high', categorical_transformer_high, categorical_features_high)
        ],
        remainder='drop',
        verbose_feature_names_out=True
    )

    print("Pipeline de pré-processamento criado com sucesso.")


# In[7]:


if 'df' in locals():
    print("\n--- Divisão de Dados Baseada em Fluxo (Group Split) e Salvamento ---")

    # 1. Obter todos os IDs de fluxo únicos
    all_flow_ids = df['flow_id'].unique()

    # 2. Dividir os IDs de fluxo em treino (70%) e teste (30%)
    train_flow_ids, test_flow_ids = train_test_split(
        all_flow_ids,
        test_size=0.3,
        random_state=42
    )

    # 3. Criar os DataFrames de treino e teste com base nos IDs de fluxo
    train_df = df[df['flow_id'].isin(train_flow_ids)].copy()
    test_df = df[df['flow_id'].isin(test_flow_ids)].copy()
    print(f"Divisão concluída: {len(train_df)} linhas para treino, {len(test_df)} para teste.")

    # 4. Separar features (X) e alvo (y) para cada conjunto
    cols_to_drop = ['Anomaly_Flag', 'Timestamp', 'flow_id']
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df['Anomaly_Flag']

    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df['Anomaly_Flag']

    print(f"Formato de X_train: {X_train.shape}, Formato de X_test: {X_test.shape}")

    # 5. Aplicar o pré-processador (definido na Célula 6)
    print("\nAplicando o pré-processador...")
    # O preprocessor é treinado APENAS com os dados de treino
    X_train_proc = preprocessor.fit_transform(X_train)
    # O conjunto de teste é APENAS transformado com o preprocessor já treinado
    X_test_proc = preprocessor.transform(X_test)
    print(f"Formato de X_train processado: {X_train_proc.shape}")
    print(f"Formato de X_test processado: {X_test_proc.shape}")

    # 6. Salvar os artefatos
    processed_data_path = '../../data/processed/cidds_dataset'
    os.makedirs(processed_data_path, exist_ok=True)

    preprocessor_path = os.path.join(processed_data_path, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\nPré-processador salvo em: {preprocessor_path}")

    train_proc_path = os.path.join(processed_data_path, 'train_processed.npz')
    test_proc_path = os.path.join(processed_data_path, 'test_processed.npz')

    np.savez_compressed(train_proc_path, X=X_train_proc, y=y_train.values)
    np.savez_compressed(test_proc_path, X=X_test_proc, y=y_test.values)

    print(f"Dados de treino processados salvos em: {train_proc_path}")
    print(f"Dados de teste processados salvos em: {test_proc_path}")


# In[ ]:




