#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score,
    balanced_accuracy_score
)
from sklearn.model_selection import train_test_split # Import que faltava
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display

# Configurações
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
pio.templates.default = "plotly_dark"

print("Bibliotecas importadas e configurações aplicadas.")


# In[2]:


# --- 1. Definição de Caminhos ---

# Caminho base onde as pastas de artefatos de modelo (random_forest, xgboost, etc.) estão
model_base_path = '..\\..\\data\\preprocessors-pipeline' 

# Caminho para o dataset de validação gerado
validation_dataset_path = '..\\..\\data\\processed\\ubuntu_server_dataset_final.csv'

print(f"Caminho base dos modelos: {model_base_path}")
print(f"Caminho do dataset de validação: {validation_dataset_path}")


# In[3]:


# --- 2. Carregamento e Limpeza Pós-Leitura do Dataset de Validação ---

print(f"Carregando dataset de validação: {validation_dataset_path}...")
df_validation = pd.read_csv(validation_dataset_path, low_memory=False)

# -------------------------------------------------------------------
# BLOCO DE LIMPEZA DE VALIDAÇÃO (Implementado por você)
# -------------------------------------------------------------------
print("\n--- Verificação e Limpeza de NaNs (Pós-Leitura) ---")
print(f"DataFrame contém NaNs ANTES do tratamento: {df_validation.isnull().values.any()}")

# Coluna específica do problema
coluna_problema = 'response_body_len'

if coluna_problema in df_validation.columns:
    if df_validation[coluna_problema].isnull().values.any():
        print(f"Tratando NaNs encontrados na coluna '{coluna_problema}'...")
        # Preenche NaNs com 0
        df_validation[coluna_problema] = df_validation[coluna_problema].fillna(0)

        # Por segurança, também substitui infinitos (se houver)
        df_validation[coluna_problema] = df_validation[coluna_problema].replace([np.inf, -np.inf], 0)

        print(f"NaNs e Infs em '{coluna_problema}' preenchidos com 0.")
    else:
        print(f"Coluna '{coluna_problema}' checada, não contém NaNs.")
else:
    print(f"Aviso: Coluna '{coluna_problema}' não encontrada.")

# Verificação final
print(f"DataFrame contém NaNs APÓS o tratamento: {df_validation.isnull().values.any()}")
if df_validation.isnull().values.any():
    print("\nATENÇÃO: NaNs ainda persistem! Colunas com NaNs:")
    print(df_validation.isnull().sum()[df_validation.isnull().sum() > 0])


# In[4]:


# --- 3. Preparação dos Dados para Validação ---

print("Separando features (X_val) e rótulos (y_val)...")

# Define as colunas de rótulo
label_cols = ['attack_cat', 'label']

# X_val: Todas as colunas, exceto os rótulos
X_val = df_validation.drop(columns=label_cols)

# y_val_text: O rótulo categórico (ex: 'Normal', 'DoS', 'Exploits')
y_val_text = df_validation['attack_cat']

# y_val_binary: O rótulo binário (0 para 'Normal', 1 para 'Attack')
# Mapeia 'Normal' -> 0 e qualquer outra coisa -> 1
y_val_binary = df_validation['attack_cat'].apply(lambda x: 0 if x == 'Normal' else 1)

print(f"X_val (features) shape: {X_val.shape}")
print(f"y_val_text (rótulos multiclasse) shape: {y_val_text.shape}")
print(f"y_val_binary (rótulos binários) shape: {y_val_binary.shape}")

print("\nVerificação de nulos nos dados prontos:")
print(f"  X_val contém nulos: {X_val.isnull().values.any()}")
print(f"  y_val_text contém nulos: {y_val_text.isnull().values.any()}")
print(f"  y_val_binary contém nulos: {y_val_binary.isnull().values.any()}")


# In[5]:


# --- 4. Modelos a Validar ---

# Dicionário mapeando Nome (Chave) para a pasta de artefatos (Valor)
model_directories = {
    'RandomForest': 'RF',
    'XGBoost': 'XGB',
    'SVM': 'SVM',
    'SelfTraining-RF': 'SelfTraining-RF',
    'SelfTraining-XGB': 'SelfTraining-XGB',
    'LabelSpreading': 'LabelSpreading',
    'IsolationForest': 'IF',
    'IsolationForestSubsampled': 'IF-Subsampled',
    'KMeans': 'KMeans',
    'MBKMeans': 'MBKMeans',
    'HDBSCAN': 'HDBSCAN',
    'DBSCAN': 'DBSCAN'
}

print("Lista de modelos para validação definida.")


# In[6]:


# --- 5. Função Auxiliar de Carregamento ---

def load_artifact(file_path):
    """
    Carrega um artefato (modelo, pipeline, etc.) de um arquivo .joblib ou .json.
    Retorna None se o arquivo não for encontrado ou se houver um erro.
    """
    if not os.path.exists(file_path):
        # print(f"  Aviso: Artefato não encontrado: {file_path}")
        return None
    try:
        if file_path.endswith('.joblib'):
            return joblib.load(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"  Erro ao carregar artefato {file_path}: {e}")
        return None

print("Função 'load_artifact' definida.")


# In[11]:


# --- 6. Definição de Grupos de Modelos e Carregamento do Encoder ---

# CORREÇÃO: Definimos quais modelos são multiclasse e quais são de anomalia
# Isso determina quais modelos precisam do encoder e quais não.

# Modelos que preveem 'attack_cat' e PRECISAM do LabelEncoder multiclasse
MODELS_MULTICLASS = [
    'RandomForest',
    'XGBoost',
    'SelfTraining-RF',
    'SelfTraining-XGB',
    'LabelSpreading',
    'SVM' # Assumindo que seu SVM é multiclasse, ajuste se não for.
]

# Modelos que preveem Anomalia (binário) e NÃO usam o encoder multiclasse
MODELS_ANOMALY = [
    'IsolationForest',
    'IsolationForestSubsampled',
    'KMeans',
    'MBKMeans',
    'HDBSCAN',
    'DBSCAN'
]

# CORREÇÃO: Carregamos o ÚNICO encoder multiclasse (le_ls.joblib) UMA VEZ.
# Usamos o caminho exato baseado no seu 'model_base_path', na pasta 'label_spreading'
# e no nome do arquivo 'le_ls.joblib' que você mostrou na imagem.

shared_multiclass_encoder = None
encoder_dir = model_directories.get('LabelSpreading') # 'label_spreading'
if encoder_dir:
    # Este é o caminho correto baseado em suas informações
    encoder_path = os.path.join(model_base_path, encoder_dir, 'ls_encoder_unsw_nb15.joblib')

    print(f"Tentando carregar o encoder compartilhado de: {encoder_path}")
    shared_multiclass_encoder = load_artifact(encoder_path)

if shared_multiclass_encoder:
    print(">>> Sucesso: LabelEncoder multiclasse compartilhado foi carregado.")
else:
    print(">>> ERRO CRÍTICO: Não foi possível carregar o 'shared_multiclass_encoder'.")
    print(">>> Modelos multiclasse (RF, XGB, etc.) podem falhar.")


# In[12]:


# --- 7. Loop Principal de Validação ---

print("\n--- Iniciando Loop de Validação dos Modelos ---")

# Dicionário para armazenar resultados (métricas)
results = {}
# Dicionário para armazenar matrizes de confusão (para plotagem)
confusion_matrices = {}

# Mapeia y_val_text para binário (para modelos não supervisionados)
# 'Normal' -> 0, 'Attack' (qualquer tipo) -> 1
y_val_binary_mapped = y_val_text.apply(lambda x: 0 if x == 'Normal' else 1).values

# Iterar sobre cada modelo definido no dicionário
for model_key, model_dir in model_directories.items():
    model_name = model_key # 'RandomForest', 'XGBoost', etc.
    print(f"\nValidando: {model_name}...")

    try:
        # 1. Definir caminho para os artefatos do modelo atual
        artifacts_path = os.path.join(model_base_path, model_dir)
        if not os.path.isdir(artifacts_path):
            print(f"  Aviso: Diretório de artefatos não encontrado, pulando: {artifacts_path}")
            continue

        # 2. Carregar artefatos necessários
        pipeline = load_artifact(f'{artifacts_path}\\pipeline_{model_dir}.joblib')
        model = load_artifact(f'{artifacts_path}\\model_{model_dir}.joblib')
        preprocessor = load_artifact(f'{artifacts_path}\\preprocessor_{model_dir}.joblib')
        threshold = load_artifact(f'{artifacts_path}\\threshold_{model_dir}.json')

        # 3. Gerar Predições

        y_pred_encoded = None
        y_pred_text = None

        # Lógica de predição (Pipeline vs. Manual)
        print("##################################")
        print(model_dir)
        if pipeline:
            # Se um pipeline completo foi salvo (ex: RF, XGB)
            X_val_processed = X_val # O pipeline faz o pré-processamento
            y_pred_encoded = pipeline.predict(X_val_processed)

        elif model and preprocessor:
            # Se o modelo e o preprocessor foram salvos separadamente (ex: IF, KMeans)
            X_val_processed = preprocessor.transform(X_val)

            # Lógica específica para modelos não supervisionados
            if 'IsolationForest' in model_name:
                y_pred_encoded = model.predict(X_val_processed)
            elif 'KMeans' in model_name or 'MBKMeans' in model_name:
                distances = model.transform(X_val_processed)
                anomaly_scores = np.min(distances, axis=1)
                # Usa o threshold salvo
                y_pred_encoded = (anomaly_scores > threshold['threshold']).astype(int)
            else:
                # Fallback para outros modelos
                y_pred_encoded = model.predict(X_val_processed)
        else:
            print(f"  Aviso: Artefatos insuficientes para '{model_name}'. Pulando.")
            continue

        # 4. Decodificar/Mapear Predições para Texto ('Normal', 'Attack', 'DoS', etc.)

        if model_name in MODELS_MULTICLASS:
            # Estes modelos preveem 'attack_cat' e precisam do encoder
            print("  Tipo: Classificador Multiclasse. Usando 'shared_multiclass_encoder'...")
            if shared_multiclass_encoder:
                # Esta linha usará o encoder carregado (shared_multiclass_encoder)
                # e não a variável 'encoder' local (que estaria None)
                y_pred_text = shared_multiclass_encoder.inverse_transform(y_pred_encoded)
            else:
                print(f"  ERRO: Modelo '{model_name}' é multiclasse, mas o 'shared_multiclass_encoder' não foi carregado.")
                continue

        elif model_name in MODELS_ANOMALY:
            # Estes modelos preveem anomalia (binário)
            print("  Tipo: Detector de Anomalia. Mapeando predição binária...")

            # Lógica de mapeamento (ajuste conforme necessário)
            if 'IsolationForest' in model_name:
                # IF: 1 é 'Normal', -1 é 'Attack'
                y_pred_text = np.where(y_pred_encoded == 1, 'Normal', 'Attack')
            elif 'KMeans' in model_name or 'MBKMeans' in model_name:
                # KMeans/MBK: 0 é 'Normal' (abaixo do threshold), 1 é 'Attack' (acima)
                y_pred_text = np.where(y_pred_encoded == 0, 'Normal', 'Attack')
            elif 'HDBSCAN' in model_name or 'DBSCAN' in model_name:
                # DBSCAN/HDBSCAN: -1 é 'Attack' (outlier), 0+ é 'Normal' (cluster)
                y_pred_text = np.where(y_pred_encoded == -1, 'Attack', 'Normal')
            else:
                # Fallback genérico (pode estar errado, mas é um chute)
                y_pred_text = np.where(y_pred_encoded == 0, 'Normal', 'Attack')

        else:
            print(f"  Aviso: Modelo {model_name} não classificado (MULTICLASS ou ANOMALY). Pulando.")
            continue

        # **** CORREÇÃO TERMINA AQUI ****

        # 5. Calcular Métricas
        print(f"Calculando métricas para: {model_name}")

        # Usamos o y_val_text original como 'true'
        # e o y_pred_text decodificado como 'pred'

        # Métricas multiclasse
        acc = accuracy_score(y_val_text, y_pred_text)
        bal_acc = balanced_accuracy_score(y_val_text, y_pred_text)

        # Métricas binárias (mapeando "Attack" vs "Normal")
        y_true_binary = (y_val_text != 'Normal').astype(int)
        y_pred_binary = (y_pred_text != 'Normal').astype(int)

        f1_bin = f1_score(y_true_binary, y_pred_binary)
        recall_bin = recall_score(y_true_binary, y_pred_binary)
        precision_bin = precision_score(y_true_binary, y_pred_binary)

        results[model_name] = {
            'Accuracy': acc,
            'Balanced Accuracy': bal_acc,
            'F1 Score (Binary)': f1_bin,
            'Recall (Binary)': recall_bin,
            'Precision (Binary)': precision_bin
        }

        # Gerar Matriz de Confusão
        # Obter todos os rótulos únicos de ambos os sets (true e pred)
        labels = sorted(list(set(y_val_text) | set(y_pred_text)))
        cm = confusion_matrix(y_val_text, y_pred_text, labels=labels)
        confusion_matrices[model_name] = (cm, labels)

    except Exception as e:
        print(f"  ERRO ao processar {model_name}. Pulando. Detalhe: {e}")
        # import traceback
        # traceback.print_exc() # Descomente para debug completo

print("\n--- Validação de todos os modelos concluída! ---")


# In[9]:


# --- 8. Exibição dos Resultados ---

print("\n--- Tabela de Resultados Comparativos ---")

if results:
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values(by='F1 Score (Binary)', ascending=False)

    # Formatação para melhor visualização
    styled_results = df_results.style.format({
        'Accuracy': '{:.2%}',
        'Balanced Accuracy': '{:.2%}',
        'F1 Score (Binary)': '{:.4f}',
        'Recall (Binary)': '{:.4f}',
        'Precision (Binary)': '{:.4f}'
    }).background_gradient(
        cmap='viridis', subset=['F1 Score (Binary)', 'Recall (Binary)', 'Balanced Accuracy']
    )

    display(styled_results)

    # --- 9. Plotagem das Matrizes de Confusão ---
    print("\n--- Matrizes de Confusão (Interativas) ---")

    for model_name, (cm, labels) in confusion_matrices.items():

        # Texto do hover (z)
        z_text = [[str(y) for y in x] for x in cm]

        fig = ff.create_annotated_heatmap(
            z=cm,
            x=list(labels),
            y=list(labels),
            annotation_text=z_text,
            colorscale='Viridis',
            showscale=True
        )

        fig.update_layout(
            title=f'Matriz de Confusão: {model_name}',
            xaxis_title="Predição",
            yaxis_title="Verdadeiro",
            yaxis=dict(autorange="reversed") # Coloca o (0,0) no canto superior esquerdo
        )

        fig.show()

else:
    print("Nenhum resultado foi gerado. Verifique os erros no loop de validação.")


# In[ ]:




