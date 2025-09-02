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
        
        eeg_data = []
        timestamps = []
        
        sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[0])
        
        for node_id, attrs in sorted_nodes:
            start_time = float(attrs.get('start_time', 0))
            end_time = float(attrs.get('end_time', 0))
            timestamps.append((start_time, end_time))
            
            eeg_array = extract_eeg_from_node(attrs)
            
            if eeg_array is not None:
                eeg_data.append(eeg_array)
            else:
                eeg_data.append(None)
        
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