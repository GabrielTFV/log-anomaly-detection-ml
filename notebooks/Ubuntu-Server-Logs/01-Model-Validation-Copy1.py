# In[1]:
#!/usr/bin/env python
# coding: utf-8

# ### 01. Validação dos Modelos: Binária (Global) e Multiclasse (Supervisionada)
# 
# **Objetivo:** # 1. Avaliar TODOS os modelos na capacidade de detectar intrusão (Binário: Normal vs Attack).
# 2. Avaliar APENAS modelos supervisionados/semi na capacidade de classificar o tipo de ataque (Multiclasse).

import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, balanced_accuracy_score, classification_report
)
import plotly.figure_factory as ff
import plotly.io as pio
from IPython.display import display

# Configurações
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
pio.templates.default = "plotly_dark"

print("Bibliotecas importadas e configurações aplicadas.")


# In[2]:
# --- 1. Definição de Caminhos e Grupos ---

base_path = '..\\..\\data' # Ajuste conforme sua estrutura
model_base_path = os.path.join(base_path, 'preprocessors-pipeline') 
validation_dataset_path = os.path.join(base_path, 'ubuntu_server_dataset_FINAL.csv')

# Grupo 1: Modelos Supervisionados/Semi (Podem fazer Multiclasse)
MODELS_SUPERVISED = [
    'RandomForest', 'XGBoost', 'SVM', 
    'SelfTraining-RF', 'SelfTraining-XGB', 'LabelSpreading'
]

# Grupo 2: Modelos Não Supervisionados (Apenas Binário)
MODELS_UNSUPERVISED = [
    'IsolationForest', 'IsolationForestSubsampled', 
    'KMeans', 'MBKMeans', 'HDBSCAN', 'DBSCAN'
]

# Mapeamento de pastas
model_directories = {
    'RandomForest': 'random_forest',
    'XGBoost': 'xgboost',
    'SVM': 'svm',
    'SelfTraining-RF': 'self_training_rf',
    'SelfTraining-XGB': 'self_training_xgb',
    'LabelSpreading': 'label_spreading',
    'IsolationForest': 'isolation_forest',
    'IsolationForestSubsampled': 'isolation_forest_subsampled',
    'KMeans': 'kmeans',
    'MBKMeans': 'mbkmeans',
    'HDBSCAN': 'hdbscan',
    'DBSCAN': 'dbscan'
}


# In[3]:
# --- 2. Carregamento e Preparação dos Dados ---

print(f"Carregando dataset de validação: {validation_dataset_path}...")
df_validation = pd.read_csv(validation_dataset_path, low_memory=False)

# --- LIMPEZA DE SEGURANÇA (Pós-Leitura) ---
coluna_problema = 'response_body_len'
if coluna_problema in df_validation.columns:
    df_validation[coluna_problema] = df_validation[coluna_problema].fillna(0).replace([np.inf, -np.inf], 0)

# Separar X (Features) e y (Labels)
label_cols = ['attack_cat', 'Label']
X_val = df_validation.drop(columns=label_cols, errors='ignore')

# GABARITOS (Ground Truth)
# 1. Gabarito Multiclasse (Texto: 'Normal', 'DoS', etc)
y_val_text = df_validation['attack_cat']

# 2. Gabarito Binário (Numérico: 0=Normal, 1=Ataque)
# Mapeia 'Normal' -> 0 e qualquer outra coisa -> 1
y_val_binary = y_val_text.apply(lambda x: 0 if x == 'Normal' else 1).values

print(f"Dados prontos. Total de amostras: {len(df_validation)}")


# In[4]:
# --- 3. Função Auxiliar de Carregamento ---

def load_artifact(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        if file_path.endswith('.joblib'):
            return joblib.load(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"  Erro ao carregar {file_path}: {e}")
        return None


# In[5]:
# --- 4. ETAPA 1: VALIDAÇÃO BINÁRIA (TODOS OS MODELOS) ---
# Aqui normalizamos tudo para 0 (Normal) e 1 (Ataque) para comparação justa.

print("\n=== INICIANDO VALIDAÇÃO BINÁRIA (Detecção de Intrusão) ===")
results_binary = {}
cm_binary = {}

for model_name, model_dir in model_directories.items():
    print(f"\nValidando (Binário): {model_name}...")
    
    # 1. Carregar Artefatos
    path = os.path.join(model_base_path, model_dir)
    pipeline = load_artifact(os.path.join(path, f'pipeline_{model_dir}.joblib'))
    model = load_artifact(os.path.join(path, f'model_{model_dir}.joblib'))
    preprocessor = load_artifact(os.path.join(path, f'preprocessor_{model_dir}.joblib'))
    threshold_data = load_artifact(os.path.join(path, f'threshold_{model_dir}.json'))
    
    y_pred_bin = None
    
    try:
        # 2. Gerar Predições Brutas e Converter para Binário (0/1)
        
        # CASO A: Modelos Supervisionados (Pipeline ou Modelo+Encoder)
        if model_name in MODELS_SUPERVISED:
            # Pega a predição multiclasse e converte para binária
            y_pred_raw = None
            encoder = None
            
            # Tenta carregar encoder específico para decodificar
            enc_name = 'le_ls.joblib' if model_name == 'LabelSpreading' else f'le_{model_dir}.joblib'
            encoder = load_artifact(os.path.join(path, enc_name))
            
            if pipeline:
                y_pred_raw = pipeline.predict(X_val)
            elif model and preprocessor:
                y_pred_raw = model.predict(preprocessor.transform(X_val))
            
            if y_pred_raw is not None and encoder is not None:
                # Decodifica para texto (ex: 'DoS', 'Normal')
                try:
                    y_text = encoder.inverse_transform(y_pred_raw)
                    # Converte para 0/1 (0=Normal, 1=Qualquer outra coisa)
                    y_pred_bin = np.where(y_text == 'Normal', 0, 1)
                except Exception as e:
                    print(f"  Erro na decodificação binária: {e}")
            else:
                print("  Artefatos incompletos para supervisionado.")
                continue

        # CASO B: Modelos Não Supervisionados (Lógica de Anomalia)
        elif model_name in MODELS_UNSUPERVISED:
            X_proc = preprocessor.transform(X_val) if preprocessor else X_val
            
            if 'IsolationForest' in model_name:
                # IF retorna: 1 (Normal), -1 (Anomalia)
                # Converter para: 0 (Normal), 1 (Ataque)
                preds = model.predict(X_proc)
                y_pred_bin = np.where(preds == 1, 0, 1)
                
            elif 'KMeans' in model_name or 'MBKMeans' in model_name:
                # Baseado em distância e threshold
                if threshold_data:
                    distances = model.transform(X_proc)
                    min_dist = np.min(distances, axis=1)
                    # Acima do threshold = Ataque (1), Abaixo = Normal (0)
                    y_pred_bin = (min_dist > threshold_data['threshold']).astype(int)
            
            elif 'HDBSCAN' in model_name or 'DBSCAN' in model_name:
                 # -1 é ruído (Ataque), outros clusters são normais
                 # Aviso: DBSCAN precisa de re-fit ou lógica complexa para prever novos dados,
                 # assumindo aqui que 'model' suporta predict ou fit_predict em validação
                 if hasattr(model, 'predict'): # HDBSCAN moderno ou wrapper
                     preds = model.predict(X_proc)
                     y_pred_bin = np.where(preds == -1, 1, 0)
                 elif hasattr(model, 'fit_predict'): # DBSCAN padrão do sklearn não tem predict
                     # DBSCAN transductivo, teria que rodar fit_predict no dataset todo (lento)
                     print("  DBSCAN padrão não suporta .predict() em novos dados. Pulando.")
                     continue

        # 3. Calcular Métricas Binárias
        if y_pred_bin is not None:
            acc = accuracy_score(y_val_binary, y_pred_bin)
            f1 = f1_score(y_val_binary, y_pred_bin)
            prec = precision_score(y_val_binary, y_pred_bin)
            rec = recall_score(y_val_binary, y_pred_bin)
            
            results_binary[model_name] = {
                'Accuracy': acc, 'F1-Score': f1, 'Precision': prec, 'Recall': rec
            }
            cm_binary[model_name] = confusion_matrix(y_val_binary, y_pred_bin)
            print(f"  > F1-Score: {f1:.4f} | Recall: {rec:.4f}")

    except Exception as e:
        print(f"  ERRO CRÍTICO ao validar {model_name}: {e}")

# Exibir Resultados Binários
print("\n--- RESULTADOS GERAIS: DETECÇÃO DE INTRUSÃO (BINÁRIO) ---")
if results_binary:
    df_bin = pd.DataFrame(results_binary).T.sort_values(by='F1-Score', ascending=False)
    display(df_bin.style.background_gradient(cmap='viridis', subset=['F1-Score', 'Recall']))
else:
    print("Nenhum resultado binário gerado.")


# In[6]:
# --- 5. ETAPA 2: VALIDAÇÃO MULTICLASSE (APENAS SUPERVISIONADOS) ---
# Aqui avaliamos se o modelo sabe diferenciar 'DoS' de 'Fuzzers', etc.

print("\n=== INICIANDO VALIDAÇÃO MULTICLASSE (Classificação de Ataques) ===")
results_multi = {}
cm_multi = {}

for model_name in MODELS_SUPERVISED:
    # Verifica se o modelo existe no dicionário de diretórios
    if model_name not in model_directories: continue
    
    print(f"\nValidando (Multiclasse): {model_name}...")
    model_dir = model_directories[model_name]
    path = os.path.join(model_base_path, model_dir)
    
    # Carregar artefatos (Pipeline e Encoder são os cruciais aqui)
    pipeline = load_artifact(os.path.join(path, f'pipeline_{model_dir}.joblib'))
    # Lógica para encoder específico (reforçando correção anterior)
    enc_filename = 'le_ls.joblib' if model_name == 'LabelSpreading' else f'le_{model_dir}.joblib'
    encoder = load_artifact(os.path.join(path, enc_filename))
    
    try:
        if pipeline and encoder:
            # 1. Predição Numérica
            y_pred_enc = pipeline.predict(X_val)
            
            # 2. Decodificação para Texto (Usando encoder específico salvo no treino)
            # Isso evita o erro de "unseen labels"
            try:
                y_pred_text = encoder.inverse_transform(y_pred_enc)
            except ValueError as ve:
                print(f"  Erro de Labels Desconhecidos no Encoder: {ve}")
                print("  (O modelo previu uma classe que o encoder não conhece ou vice-versa)")
                continue

            # 3. Métricas Multiclasse
            acc = accuracy_score(y_val_text, y_pred_text)
            bal_acc = balanced_accuracy_score(y_val_text, y_pred_text)
            # F1 Weighted para ter uma média ponderada das classes
            f1_w = f1_score(y_val_text, y_pred_text, average='weighted')
            
            results_multi[model_name] = {
                'Accuracy': acc, 'Balanced Acc': bal_acc, 'F1-Weighted': f1_w
            }
            
            # Matriz de Confusão com Labels de Texto
            labels = sorted(list(set(y_val_text) | set(y_pred_text)))
            cm_multi[model_name] = (confusion_matrix(y_val_text, y_pred_text, labels=labels), labels)
            
            print(f"  > Accuracy: {acc:.2%} | Balanced Acc: {bal_acc:.2%}")
        else:
            print("  Pipeline ou Encoder não encontrados. Pulando.")

    except Exception as e:
        print(f"  Erro ao validar multiclasse {model_name}: {e}")

# Exibir Resultados Multiclasse
print("\n--- RESULTADOS: CLASSIFICAÇÃO DE TIPOS DE ATAQUE ---")
if results_multi:
    df_multi = pd.DataFrame(results_multi).T.sort_values(by='Balanced Acc', ascending=False)
    display(df_multi.style.background_gradient(cmap='magma'))
else:
    print("Nenhum resultado multiclasse gerado.")


# In[7]:
# --- 6. Visualização: Matrizes de Confusão ---

# Plotar Binárias (Top 3 Melhores)
if results_binary:
    top_bin = sorted(results_binary, key=lambda x: results_binary[x]['F1-Score'], reverse=True)[:3]
    print(f"\nTop 3 Modelos Binários (F1-Score): {top_bin}")
    
    for m in top_bin:
        cm = cm_binary[m]
        fig = ff.create_annotated_heatmap(
            z=cm, x=['Normal', 'Attack'], y=['Normal', 'Attack'], 
            colorscale='Viridis', annotation_text=[[str(y) for y in x] for x in cm]
        )
        fig.update_layout(title=f"Matriz Binária: {m} (Normal vs Attack)")
        fig.show()

# Plotar Multiclasse (Todos que rodaram)
if results_multi:
    print("\nMatrizes Multiclasse (Detalhado):")
    for m, (cm, labels) in cm_multi.items():
        fig = ff.create_annotated_heatmap(
            z=cm, x=list(labels), y=list(labels), 
            colorscale='Magma', annotation_text=[[str(y) for y in x] for x in cm]
        )
        fig.update_layout(title=f"Matriz Multiclasse: {m}", xaxis_title="Predito", yaxis_title="Real")
        fig.show()