import numpy as np
import networkx as nx

def extract_eeg_from_node(node_attrs):
    """
    Extrai dados EEG de um nó do grafo e converte para array numpy.
    
    Args:
        node_attrs (dict): Atributos do nó do grafo
        
    Returns:
        np.ndarray or None: Array com features EEG ou None se não disponível
    """
    eeg_data_str = node_attrs.get('eeg_data', 'NA')
    
    if eeg_data_str == 'NA' or not eeg_data_str:
        return None
    
    try:
        # Converter string separada por vírgulas de volta para array
        eeg_values = [float(x) for x in eeg_data_str.split(',')]
        return np.array(eeg_values)
    except (ValueError, AttributeError):
        return None

def load_graph_with_eeg(graph_path):
    """
    Carrega grafo e extrai todos os dados EEG dos nós.
    
    Args:
        graph_path (str): Caminho para o arquivo .gml
        
    Returns:
        tuple: (graph, eeg_matrix, timestamps)
            - graph: objeto NetworkX
            - eeg_matrix: matriz numpy (n_events, n_features) ou None
            - timestamps: lista de tuplas (start_time, end_time)
    """
    try:
        G = nx.read_gml(graph_path)
        
        # Coletar dados EEG e timestamps de todos os nós
        eeg_data = []
        timestamps = []
        
        # Ordenar nós por ID
        sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[0])
        
        for node_id, attrs in sorted_nodes:
            # Extrair timestamps
            start_time = float(attrs.get('start_time', 0))
            end_time = float(attrs.get('end_time', 0))
            timestamps.append((start_time, end_time))
            
            # Extrair dados EEG
            eeg_array = extract_eeg_from_node(attrs)
            
            if eeg_array is not None:
                eeg_data.append(eeg_array)
            else:
                # Se não há dados EEG, adicionar vetor de zeros ou None
                eeg_data.append(None)
        
        # Criar matriz EEG se há dados válidos
        valid_eeg_data = [data for data in eeg_data if data is not None]
        
        if valid_eeg_data:
            eeg_matrix = np.vstack(valid_eeg_data)
            print(f"Carregado: {len(valid_eeg_data)} eventos com dados EEG, {eeg_matrix.shape[1]} features")
        else:
            eeg_matrix = None
            print("Nenhum dado EEG encontrado no grafo")
        
        return G, eeg_matrix, timestamps
        
    except Exception as e:
        print(f"Erro ao carregar grafo {graph_path}: {e}")
        return None, None, None

def get_eeg_statistics(graph_path):
    """
    Calcula estatísticas dos dados EEG em um grafo.
    
    Args:
        graph_path (str): Caminho para o arquivo .gml
        
    Returns:
        dict: Estatísticas dos dados EEG
    """
    G, eeg_matrix, timestamps = load_graph_with_eeg(graph_path)
    
    if eeg_matrix is None:
        return {'status': 'no_eeg_data'}
    
    stats = {
        'status': 'success',
        'n_events': eeg_matrix.shape[0],
        'n_features': eeg_matrix.shape[1],
        'total_duration_ms': timestamps[-1][1] - timestamps[0][0] if timestamps else 0,
        'mean_features': np.mean(eeg_matrix, axis=0),
        'std_features': np.std(eeg_matrix, axis=0),
        'min_features': np.min(eeg_matrix, axis=0),
        'max_features': np.max(eeg_matrix, axis=0)
    }
    
    return stats

def filter_nodes_by_eeg_pattern(graph_path, feature_indices, threshold_func):
    """
    Filtra nós do grafo baseado em padrões dos dados EEG.
    
    Args:
        graph_path (str): Caminho para o arquivo .gml
        feature_indices (list): Índices das features EEG a considerar
        threshold_func (callable): Função que retorna True/False dado um vetor EEG
        
    Returns:
        list: Lista de node_ids que atendem ao critério
    """
    G, eeg_matrix, timestamps = load_graph_with_eeg(graph_path)
    
    if eeg_matrix is None:
        return []
    
    filtered_nodes = []
    sorted_nodes = sorted(G.nodes(), key=int)
    
    for i, node_id in enumerate(sorted_nodes):
        if i < len(eeg_matrix):
            eeg_features = eeg_matrix[i, feature_indices]
            
            if threshold_func(eeg_features):
                filtered_nodes.append(node_id)
    
    return filtered_nodes

def analyze_eeg_temporal_patterns(graph_path, window_size=5):
    """
    Analisa padrões temporais nos dados EEG do grafo.
    
    Args:
        graph_path (str): Caminho para o arquivo .gml
        window_size (int): Tamanho da janela deslizante
        
    Returns:
        dict: Análise de padrões temporais
    """
    G, eeg_matrix, timestamps = load_graph_with_eeg(graph_path)
    
    if eeg_matrix is None or len(eeg_matrix) < window_size:
        return {'status': 'insufficient_data'}
    
    # Calcular médias móveis
    moving_averages = []
    for i in range(len(eeg_matrix) - window_size + 1):
        window = eeg_matrix[i:i+window_size]
        moving_avg = np.mean(window, axis=0)
        moving_averages.append(moving_avg)
    
    moving_averages = np.array(moving_averages)
    
    # Calcular variações
    variations = np.std(moving_averages, axis=0)
    
    results = {
        'status': 'success',
        'window_size': window_size,
        'n_windows': len(moving_averages),
        'temporal_variations': variations,
        'most_variable_features': np.argsort(variations)[-10:],  # Top 10 mais variáveis
        'least_variable_features': np.argsort(variations)[:10],   # Top 10 menos variáveis
        'moving_averages': moving_averages
    }
    
    return results

if __name__ == "__main__":
    # Teste das funcionalidades
    graph_path = 'database/graph/subject_1/session_1_trial_1.gml'
    
    print("=== TESTE DAS UTILIDADES DE GRAFO ===")
    
    # Testar carregamento
    G, eeg_matrix, timestamps = load_graph_with_eeg(graph_path)
    
    if eeg_matrix is not None:
        print(f"✓ Carregamento bem-sucedido: {eeg_matrix.shape}")
        
        # Testar estatísticas
        stats = get_eeg_statistics(graph_path)
        print(f"✓ Estatísticas: {stats['n_events']} eventos, {stats['n_features']} features")
        
        # Testar análise temporal
        temporal_analysis = analyze_eeg_temporal_patterns(graph_path, window_size=5)
        if temporal_analysis['status'] == 'success':
            print(f"✓ Análise temporal: {temporal_analysis['n_windows']} janelas analisadas")
    else:
        print("✗ Nenhum dado EEG encontrado para teste")