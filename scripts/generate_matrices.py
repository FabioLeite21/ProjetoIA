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
    Extrai features dos dados EEG - retorna os 256 valores completos.
    
    Args:
        eeg_data_str (str): String com dados EEG separados por vírgula
        
    Returns:
        list: Lista com os 256 valores EEG completos
    """
    if eeg_data_str == "NA" or not eeg_data_str:
        return [0.0] * 256  # 256 features padrão com valores zero
    
    try:
        eeg_values = [float(x) for x in eeg_data_str.split(',')]
        
        # Se temos exatamente 256 valores, retorna diretamente
        if len(eeg_values) == 256:
            return eeg_values
        
        # Se temos menos de 256 valores, preenche com zeros
        elif len(eeg_values) < 256:
            eeg_values.extend([0.0] * (256 - len(eeg_values)))
            return eeg_values
        
        # Se temos mais de 256 valores, trunca para 256
        else:
            return eeg_values[:256]
            
    except Exception as e:
        print(f"Erro ao processar dados EEG: {e}")
        return [0.0] * 256

def extract_node_features(node_data):
    """
    Extrai features de um nó do grafo.
    
    Args:
        node_data (dict): Dados do nó
        
    Returns:
        list: Lista com todas as features do nó (7 eye-tracking + 256 EEG = 263 total)
    """
    features = []
    
    # Features básicas do eye-tracking (7 features)
    eye_tracking_fields = [
        'fixation_duration',  # duração da fixação
        'pupil_x',           # tamanho da pupila X
        'pupil_y',           # tamanho da pupila Y
        'dispersion_x',      # dispersão X
        'dispersion_y',      # dispersão Y
        'start_time',        # tempo de início
        'end_time'           # tempo de fim
    ]
    
    for field in eye_tracking_fields:
        try:
            value = float(node_data.get(field, 0))
            features.append(value)
        except (ValueError, TypeError):
            features.append(0.0)
    
    # Features dos dados EEG (256 features)
    eeg_features = extract_eeg_features(node_data.get('eeg_data', 'NA'))
    features.extend(eeg_features)
    
    # Verificação final: garantir que temos exatamente 263 features
    if len(features) != 263:
        print(f"Aviso: Features extraídas = {len(features)}, esperado = 263")
        # Ajustar para 263 se necessário
        if len(features) < 263:
            features.extend([0.0] * (263 - len(features)))
        else:
            features = features[:263]
    
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
    
    # Define nomes das colunas para a matriz de features (263 total)
    feature_names = []
    
    # Eye-tracking features (7)
    eye_tracking_names = [
        'fixation_duration', 'pupil_x', 'pupil_y', 'dispersion_x', 'dispersion_y',
        'start_time', 'end_time'
    ]
    feature_names.extend(eye_tracking_names)
    
    # EEG features (256)
    eeg_names = [f'eeg_{i+1:03d}' for i in range(256)]
    feature_names.extend(eeg_names)
    
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
    print(f"• {total_processed} grafos processados")
    print(f"• {total_success} matrizes geradas com sucesso")
    print(f"• {total_errors} erros encontrados")
    if total_processed > 0:
        print(f"• Taxa de sucesso: {total_success/total_processed*100:.1f}%")
    
    print("\nTipos de arquivos gerados:")
    print("• *_adjacency_matrix.csv - Matrizes de adjacência (0s e 1s)")
    print("• *_feature_matrix.csv - Matrizes de features (eye-tracking + EEG)")
    print("\nProcessamento concluído!")

if __name__ == "__main__":
    generate_all_matrices()