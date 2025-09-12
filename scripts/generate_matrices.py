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
from multiprocessing import Pool, cpu_count
import time

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

def extract_eeg_features(eeg_data_str):
    """
    Extrai features dos dados EEG - retorna os 310 valores completos (62 canais × 5 bandas).
    Otimizado com NumPy para melhor performance.
    
    Args:
        eeg_data_str (str): String com dados EEG separados por vírgula
        
    Returns:
        np.ndarray: Array NumPy com os 310 valores EEG completos
    """
    if eeg_data_str == "NA" or not eeg_data_str:
        return np.zeros(310, dtype=np.float64)
    
    try:
        eeg_values = np.fromiter(
            (float(x) for x in eeg_data_str.split(',')),
            dtype=np.float64
        )
        
        if len(eeg_values) == 310:
            return eeg_values
        
        elif len(eeg_values) < 310:
            result = np.zeros(310, dtype=np.float64)
            result[:len(eeg_values)] = eeg_values
            return result
        
        else:
            return eeg_values[:310]
            
    except Exception as e:
        print(f"Erro ao processar dados EEG: {e}")
        return np.zeros(310, dtype=np.float64)

def extract_node_features(node_data):
    """
    Extrai features de um nó do grafo.
    Otimizado com pré-alocação NumPy para melhor performance.
    
    Args:
        node_data (dict): Dados do nó
        
    Returns:
        np.ndarray: Array NumPy com todas as features do nó (7 eye-tracking + 310 EEG = 317 total)
    """
    features = np.zeros(317, dtype=np.float64)
    
    eye_tracking_fields = [
        'fixation_duration',
        'pupil_x',
        'pupil_y',
        'dispersion_x',
        'dispersion_y',
        'start_time',
        'end_time'
    ]
    
    for i, field in enumerate(eye_tracking_fields):
        try:
            features[i] = float(node_data.get(field, 0))
        except (ValueError, TypeError):
            features[i] = 0.0
    
    eeg_features = extract_eeg_features(node_data.get('eeg_data', 'NA'))
    features[7:317] = eeg_features
    
    return features

def create_adjacency_matrix(graph):
    """
    Cria matriz de adjacência a partir do grafo usando NetworkX.
    
    Args:
        graph (networkx.DiGraph): Grafo dirigido
        
    Returns:
        np.ndarray: Matriz de adjacência
    """
    adj_matrix = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    return adj_matrix

def create_feature_matrix(graph):
    """
    Cria matriz de features a partir do grafo.
    Otimizado com pré-alocação NumPy para melhor performance.
    
    Args:
        graph (networkx.DiGraph): Grafo dirigido
        
    Returns:
        np.ndarray: Matriz de features
    """
    nodes = sorted(graph.nodes())
    num_nodes = len(nodes)
    
    feature_matrix = np.zeros((num_nodes, 317), dtype=np.float64)
    
    for i, node in enumerate(nodes):
        node_data = graph.nodes[node]
        feature_matrix[i] = extract_node_features(node_data)
    
    return feature_matrix

def process_single_graph(gml_file_path):
    """
    Processa um único grafo e gera suas matrizes.
    
    Args:
        gml_file_path (str): Caminho para o arquivo GML
        
    Returns:
        tuple: (matriz_adjacencia, matriz_features, sucesso)
    """
    try:
        G = nx.read_gml(gml_file_path)
        
        adj_matrix = create_adjacency_matrix(G)
        feature_matrix = create_feature_matrix(G)
        
        return adj_matrix, feature_matrix, True
    
    except Exception as e:
        print(f"    Erro ao processar {os.path.basename(gml_file_path)}: {str(e)[:50]}...")
        return None, None, False

def process_and_save_single_graph(args):
    """
    Processa um único grafo e salva suas matrizes para processamento paralelo.
    
    Args:
        args (tuple): (gml_file_path, output_base_path, skip_existing)
        
    Returns:
        tuple: (arquivo_base, sucesso, num_nodes, tempo_processamento)
    """
    gml_file_path, output_base_path, skip_existing = args
    
    start_time = time.time()
    
    try:
        if skip_existing:
            adj_path = output_base_path + "_adjacency_matrix.csv"
            features_path = output_base_path + "_feature_matrix.csv"
            
            if os.path.exists(adj_path) and os.path.exists(features_path):
                gml_time = os.path.getmtime(gml_file_path)
                adj_time = os.path.getmtime(adj_path)
                features_time = os.path.getmtime(features_path)
                
                if adj_time > gml_time and features_time > gml_time:
                    return os.path.basename(gml_file_path), True, 0, 0.0
        
        G = nx.read_gml(gml_file_path)
        num_nodes = G.number_of_nodes()
        
        adj_matrix = create_adjacency_matrix(G)
        feature_matrix = create_feature_matrix(G)
        
        save_matrices(adj_matrix, feature_matrix, output_base_path)
        
        processing_time = time.time() - start_time
        return os.path.basename(gml_file_path), True, num_nodes, processing_time
    
    except Exception as e:
        error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
        processing_time = time.time() - start_time
        return os.path.basename(gml_file_path), False, 0, processing_time

def save_matrices(adj_matrix, feature_matrix, output_base_path):
    """
    Salva as matrizes em arquivos CSV.
    
    Args:
        adj_matrix (np.ndarray): Matriz de adjacência
        feature_matrix (np.ndarray): Matriz de features
        output_base_path (str): Caminho base para salvar os arquivos
    """
    adj_path = output_base_path + "_adjacency_matrix.csv"
    np.savetxt(adj_path, adj_matrix, delimiter=',', fmt='%d')
    
    features_path = output_base_path + "_feature_matrix.csv"
    
    feature_names = []
    
    eye_tracking_names = [
        'fixation_duration', 'pupil_x', 'pupil_y', 'dispersion_x', 'dispersion_y',
        'start_time', 'end_time'
    ]
    feature_names.extend(eye_tracking_names)
    
    eeg_names = [f'eeg_{i+1:03d}' for i in range(310)]
    feature_names.extend(eeg_names)
    
    df_features = pd.DataFrame(feature_matrix, columns=feature_names)
    df_features.to_csv(features_path, index=False)

def generate_all_matrices(num_processes=None, skip_existing=False):
    """
    Gera matrizes de adjacência e features para todos os grafos usando processamento paralelo.
    Por padrão substitui todas as matrizes existentes.
    
    Args:
        num_processes (int): Número de processos paralelos. Se None, usa cpu_count()
        skip_existing (bool): Se True, pula arquivos que já existem e são mais novos.
                             Por padrão False (sempre regenera as matrizes)
    """
    start_total_time = time.time()
    
    print("Iniciando geracao paralela de matrizes de adjacencia e features...")
    
    if num_processes is None:
        num_processes = min(cpu_count(), 8)
    
    print(f"Usando {num_processes} processos paralelos")
    if skip_existing:
        print("Modo: Skip arquivos existentes mais novos")
    else:
        print("Modo: SUBSTITUIR todas as matrizes existentes")
    print()
    
    graph_base_dir = os.path.join(project_root, 'database', 'graph')
    
    all_tasks = []
    
    for subject_dir in sorted(os.listdir(graph_base_dir)):
        subject_path = os.path.join(graph_base_dir, subject_dir)
        
        if not os.path.isdir(subject_path):
            continue
        
        gml_pattern = os.path.join(subject_path, "*.gml")
        gml_files = sorted(glob.glob(gml_pattern))
        
        for gml_file in gml_files:
            base_name = os.path.splitext(os.path.basename(gml_file))[0]
            output_base_path = os.path.join(subject_path, base_name)
            
            all_tasks.append((gml_file, output_base_path, skip_existing))
    
    total_files = len(all_tasks)
    print(f"Total de {total_files} grafos para processar...\n")
    
    if total_files == 0:
        print("Nenhum arquivo GML encontrado!")
        return
    
    successful_results = []
    failed_results = []
    skipped_count = 0
    total_nodes = 0
    
    print("Processando em paralelo...")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_and_save_single_graph, all_tasks)
    
    for filename, success, num_nodes, proc_time in results:
        if success:
            successful_results.append((filename, num_nodes, proc_time))
            total_nodes += num_nodes
            if proc_time == 0.0:
                skipped_count += 1
        else:
            failed_results.append((filename, proc_time))
    
    total_time = time.time() - start_total_time
    
    print(f"\nRelatorio final - processamento paralelo:")
    print(f"Total de arquivos: {total_files}")
    print(f"Processados com sucesso: {len(successful_results)}")
    print(f"Erros encontrados: {len(failed_results)}")
    print(f"Arquivos pulados (ja existentes): {skipped_count}")
    if total_files > 0:
        success_rate = (len(successful_results) / total_files) * 100
        print(f"Taxa de sucesso: {success_rate:.1f}%")
    
    print(f"\nEstatisticas de performance:")
    print(f"Tempo total: {total_time:.2f}s")
    if len(successful_results) > 0:
        avg_time = sum(t for _, _, t in successful_results if t > 0) / max(1, len(successful_results) - skipped_count)
        print(f"Tempo medio por grafo: {avg_time:.3f}s")
    print(f"Total de nos processados: {total_nodes:,}")
    print(f"Processos paralelos: {num_processes}")
    
    print(f"\nTipos de arquivos gerados:")
    print(f"*_adjacency_matrix.csv - Matrizes de adjacencia (0s e 1s)")
    print(f"*_feature_matrix.csv - Matrizes de features (317 colunas: 7 eye-tracking + 310 EEG)")
    
    if failed_results:
        print(f"\nArquivos com erro:")
        for filename, _ in failed_results[:5]:
            print(f"  {filename}")
        if len(failed_results) > 5:
            print(f"  ... e mais {len(failed_results) - 5} arquivos")
    
    print(f"\nProcessamento paralelo concluido!")

def main():
    """
    Função principal para execução via linha de comando.
    
    Processa argumentos da linha de comando e executa a geração de matrizes
    com os parâmetros especificados.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Gera matrizes de adjacência e features dos grafos')
    parser.add_argument('--processes', '-p', type=int, default=None,
                        help='Número de processos paralelos (padrão: auto)')
    parser.add_argument('--skip-existing', '-s', action='store_true',
                        help='Pula arquivos que ja existem e sao mais novos que o GML')
    
    args = parser.parse_args()
    
    print(f"Executando generate_matrices.py")
    print(f"Processos: {args.processes if args.processes else 'auto'}")
    print(f"Skip existing: {'Sim' if args.skip_existing else 'Nao (substitui tudo)'}")
    print()
    
    generate_all_matrices(num_processes=args.processes, skip_existing=args.skip_existing)

if __name__ == "__main__":
    main()