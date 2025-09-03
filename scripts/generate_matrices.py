"""
Script para gerar matrizes de adjacência e matrizes de features a partir dos grafos existentes.
"""

import sys
import os
import glob
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to sys.path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

def extract_eeg_features(eeg_data_str):
    """
    Extrai features dos dados EEG.
    
    Args:
        eeg_data_str (str): String com dados EEG separados por vírgula
        
    Returns:
        list: Lista com features extraídas dos dados EEG
    """
    if eeg_data_str == "NA" or not eeg_data_str:
        return [0.0] * 7  # 7 features padrão com valores zero
    
    try:
        eeg_values = [float(x) for x in eeg_data_str.split(',')]
        
        # Extrai estatísticas dos dados EEG como features
        features = [
            np.mean(eeg_values),        # média
            np.std(eeg_values),         # desvio padrão
            np.min(eeg_values),         # valor mínimo
            np.max(eeg_values),         # valor máximo
            np.median(eeg_values),      # mediana
            np.var(eeg_values),         # variância
            len(eeg_values)             # número de medições
        ]
        
        return features
    except:
        return [0.0] * 7

def extract_node_features(node_data):
    """
    Extrai features de um nó do grafo.
    
    Args:
        node_data (dict): Dados do nó
        
    Returns:
        list: Lista com todas as features do nó
    """
    features = []
    
    # Features básicas do eye-tracking
    try:
        features.append(float(node_data.get('fixation_duration', 0)))
    except:
        features.append(0.0)
    
    try:
        features.append(float(node_data.get('pupil_x', 0)))
    except:
        features.append(0.0)
    
    try:
        features.append(float(node_data.get('pupil_y', 0)))
    except:
        features.append(0.0)
    
    try:
        features.append(float(node_data.get('dispersion_x', 0)))
    except:
        features.append(0.0)
    
    try:
        features.append(float(node_data.get('dispersion_y', 0)))
    except:
        features.append(0.0)
    
    # Features dos dados EEG
    eeg_features = extract_eeg_features(node_data.get('eeg_data', 'NA'))
    features.extend(eeg_features)
    
    return features

def create_adjacency_matrix(graph):
    """
    Cria matriz de adjacência a partir do grafo usando NetworkX.
    
    Args:
        graph (networkx.DiGraph): Grafo dirigido
        
    Returns:
        np.ndarray: Matriz de adjacência
    """
    # Usa função nativa do NetworkX para criar matriz de adjacência
    adj_matrix = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    return adj_matrix

def create_feature_matrix(graph):
    """
    Cria matriz de features a partir do grafo.
    
    Args:
        graph (networkx.DiGraph): Grafo dirigido
        
    Returns:
        np.ndarray: Matriz de features
    """
    # Obtém os nós ordenados
    nodes = sorted(graph.nodes())
    
    # Extrai features de todos os nós
    all_features = []
    for node in nodes:
        node_data = graph.nodes[node]
        features = extract_node_features(node_data)
        all_features.append(features)
    
    return np.array(all_features)

def process_single_graph(gml_file_path):
    """
    Processa um único grafo e gera suas matrizes.
    
    Args:
        gml_file_path (str): Caminho para o arquivo GML
        
    Returns:
        tuple: (matriz_adjacencia, matriz_features, sucesso)
    """
    try:
        # Carrega o grafo
        G = nx.read_gml(gml_file_path)
        
        # Cria as matrizes
        adj_matrix = create_adjacency_matrix(G)
        feature_matrix = create_feature_matrix(G)
        
        return adj_matrix, feature_matrix, True
    
    except Exception as e:
        print(f"    Erro ao processar {os.path.basename(gml_file_path)}: {str(e)[:50]}...")
        return None, None, False

def save_matrices(adj_matrix, feature_matrix, output_base_path):
    """
    Salva as matrizes em arquivos CSV.
    
    Args:
        adj_matrix (np.ndarray): Matriz de adjacência
        feature_matrix (np.ndarray): Matriz de features
        output_base_path (str): Caminho base para salvar os arquivos
    """
    # Salva matriz de adjacência
    adj_path = output_base_path + "_adjacency_matrix.csv"
    np.savetxt(adj_path, adj_matrix, delimiter=',', fmt='%d')
    
    # Salva matriz de features
    features_path = output_base_path + "_feature_matrix.csv"
    
    # Define nomes das colunas para a matriz de features
    feature_names = [
        'fixation_duration', 'pupil_x', 'pupil_y', 'dispersion_x', 'dispersion_y',
        'eeg_mean', 'eeg_std', 'eeg_min', 'eeg_max', 'eeg_median', 'eeg_var', 'eeg_count'
    ]
    
    # Cria DataFrame e salva
    df_features = pd.DataFrame(feature_matrix, columns=feature_names)
    df_features.to_csv(features_path, index=False)

def generate_all_matrices():
    """
    Gera matrizes de adjacência e features para todos os grafos.
    """
    print("Iniciando geração de matrizes de adjacência e features...\n")
    
    # Diretório base dos grafos
    graph_base_dir = os.path.join(project_root, 'database', 'graph')
    
    # Estatísticas
    total_processed = 0
    total_success = 0
    total_errors = 0
    
    # Processa todos os sujeitos
    for subject_dir in sorted(os.listdir(graph_base_dir)):
        subject_path = os.path.join(graph_base_dir, subject_dir)
        
        if not os.path.isdir(subject_path):
            continue
        
        print(f"Processando {subject_dir}...")
        
        # Encontra todos os arquivos GML do sujeito
        gml_pattern = os.path.join(subject_path, "*.gml")
        gml_files = sorted(glob.glob(gml_pattern))
        
        subject_processed = 0
        subject_success = 0
        
        for gml_file in gml_files:
            # Obtém o nome base do arquivo sem extensão
            base_name = os.path.splitext(os.path.basename(gml_file))[0]
            output_base_path = os.path.join(subject_path, base_name)
            
            # Processa o grafo
            adj_matrix, feature_matrix, success = process_single_graph(gml_file)
            
            total_processed += 1
            subject_processed += 1
            
            if success:
                # Salva as matrizes
                save_matrices(adj_matrix, feature_matrix, output_base_path)
                total_success += 1
                subject_success += 1
            else:
                total_errors += 1
        
        print(f"  {subject_success}/{subject_processed} grafos processados com sucesso")
    
    print(f"\nResumo do processamento:")
    print(f"{total_processed} grafos processados")
    print(f"{total_success} matrizes geradas com sucesso")
    print(f"{total_errors} erros encontrados")
    if total_processed > 0:
        print(f"Taxa de sucesso: {total_success/total_processed*100:.1f}%")
    
    print("\nTipos de arquivos gerados:")
    print("*_adjacency_matrix.csv - Matrizes de adjacência (0s e 1s)")
    print("*_feature_matrix.csv - Matrizes de features (eye-tracking + EEG)")
    print("\nProcessamento concluído!")

if __name__ == "__main__":
    generate_all_matrices()