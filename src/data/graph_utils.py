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


def save_graph_with_matrices(main_graph, graph_path, include_matrices_as_graphs=True):
    """
    Salva o grafo principal junto com suas matrizes (features, adjacência) como grafos auxiliares em um único arquivo .gml.
    
    Args:
        main_graph (nx.Graph): Grafo principal com os dados originais
        graph_path (str): Caminho para salvar o arquivo .gml
        include_matrices_as_graphs (bool): Se True, inclui matrizes como grafos auxiliares
        
    Returns:
        str: Caminho do arquivo salvo
    """
    try:
        import os
        
        if include_matrices_as_graphs:
            feature_matrix, feature_names = extract_feature_matrix_from_graph(main_graph)
            adj_matrix, edge_features = extract_adjacency_matrix_from_graph(main_graph, include_edge_features=True)
            
            compound_graph = nx.MultiDiGraph()
            
            compound_graph.add_nodes_from(main_graph.nodes(data=True))
            compound_graph.add_edges_from(main_graph.edges(data=True))
            
            compound_graph.graph['main_graph_nodes'] = len(main_graph.nodes())
            compound_graph.graph['main_graph_edges'] = len(main_graph.edges())
            compound_graph.graph['has_feature_matrix'] = feature_matrix is not None
            compound_graph.graph['has_adjacency_matrix'] = adj_matrix is not None
            
            if feature_matrix is not None:
                n_nodes, n_features = feature_matrix.shape
                for i in range(n_nodes):
                    feature_node_id = f"feature_node_{i}"
                    features_str = ','.join([f'{x:.6f}' for x in feature_matrix[i]])
                    compound_graph.add_node(feature_node_id, 
                                          node_type="feature_matrix",
                                          original_node_id=str(i),
                                          features=features_str,
                                          n_features=str(n_features))
                
                if feature_names:
                    compound_graph.graph['feature_names'] = ','.join(feature_names)
            
            if adj_matrix is not None:
                n_nodes = adj_matrix.shape[0]
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if adj_matrix[i, j] > 0:
                            adj_edge_id_source = f"adj_node_{i}"
                            adj_edge_id_target = f"adj_node_{j}"
                            
                            if adj_edge_id_source not in compound_graph:
                                compound_graph.add_node(adj_edge_id_source, 
                                                      node_type="adjacency_matrix",
                                                      matrix_row=str(i))
                            if adj_edge_id_target not in compound_graph:
                                compound_graph.add_node(adj_edge_id_target,
                                                      node_type="adjacency_matrix", 
                                                      matrix_row=str(j))
                            
                            edge_attrs = {"edge_type": "adjacency_connection", 
                                        "weight": str(adj_matrix[i, j])}
                            
                            if edge_features and (i, j) in edge_features:
                                edge_attrs.update({k: str(v) for k, v in edge_features[(i, j)].items()})
                            
                            compound_graph.add_edge(adj_edge_id_source, adj_edge_id_target, **edge_attrs)
            
            nx.write_gml(compound_graph, graph_path)
        else:
            nx.write_gml(main_graph, graph_path)
        
        print(f"Grafo salvo: {graph_path}")
        return graph_path
        
    except Exception as e:
        print(f"Erro ao salvar grafo {graph_path}: {e}")
        return None

def extract_feature_matrix_from_graph(graph):
    """
    Extrai matriz de features de um objeto grafo NetworkX.
    
    Args:
        graph (nx.Graph): Grafo NetworkX
        
    Returns:
        tuple: (feature_matrix, feature_names)
    """
    try:
        if not graph.nodes():
            return None, None
        
        sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[0])
        
        features_list = []
        feature_names = []
        
        for i, (node_id, attrs) in enumerate(sorted_nodes):
            node_features = []
            
            if i == 0:
                feature_names.extend([
                    'start_time', 'end_time', 'fixation_duration',
                    'pupil_x', 'pupil_y', 'dispersion_x', 'dispersion_y'
                ])
            
            eye_features = [
                float(attrs.get('start_time', 0)),
                float(attrs.get('end_time', 0)),
                float(attrs.get('fixation_duration', 0)),
                float(attrs.get('pupil_x', 0)),
                float(attrs.get('pupil_y', 0)),
                float(attrs.get('dispersion_x', 0)),
                float(attrs.get('dispersion_y', 0))
            ]
            node_features.extend(eye_features)
            
            eeg_array = extract_eeg_from_node(attrs)
            if eeg_array is not None:
                if i == 0:
                    eeg_names = [f'eeg_feature_{j}' for j in range(len(eeg_array))]
                    feature_names.extend(eeg_names)
                
                node_features.extend(eeg_array.tolist())
            else:
                if i == 0:
                    n_eeg_features = 248
                    eeg_names = [f'eeg_feature_{j}' for j in range(n_eeg_features)]
                    feature_names.extend(eeg_names)
                    node_features.extend([0.0] * n_eeg_features)
                else:
                    n_eeg_features = len(feature_names) - 7
                    node_features.extend([0.0] * n_eeg_features)
            
            features_list.append(node_features)
        
        feature_matrix = np.array(features_list)
        
        return feature_matrix, feature_names
        
    except Exception as e:
        print(f"Erro ao extrair features do grafo: {e}")
        return None, None

def extract_adjacency_matrix_from_graph(graph, include_edge_features=False):
    """
    Extrai matriz de adjacência de um objeto grafo NetworkX.
    
    Args:
        graph (nx.Graph): Grafo NetworkX
        include_edge_features (bool): Se True, inclui features das arestas
        
    Returns:
        tuple: (adjacency_matrix, edge_features_dict)
    """
    try:
        if not graph.nodes():
            return None, None
        
        node_ids = sorted(graph.nodes(), key=int)
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        n_nodes = len(node_ids)
        
        adj_matrix = np.zeros((n_nodes, n_nodes))
        edge_features = {} if include_edge_features else None
        
        for edge in graph.edges(data=True):
            source, target, attrs = edge
            
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]
            
            adj_matrix[source_idx, target_idx] = 1
            
            if not graph.is_directed():
                adj_matrix[target_idx, source_idx] = 1
            
            if include_edge_features:
                edge_key = (source_idx, target_idx)
                edge_features[edge_key] = {
                    'saccade_duration': float(attrs.get('saccade_duration', 0)),
                    'saccade_amplitude': float(attrs.get('saccade_amplitude', 0))
                }
        
        return adj_matrix, edge_features
        
    except Exception as e:
        print(f"Erro ao extrair adjacência do grafo: {e}")
        return None, None

def save_separate_gml_files(main_graph, graph_path):
    """
    Salva o grafo principal e suas matrizes em arquivos .gml separados.
    
    Args:
        main_graph (nx.Graph): Grafo principal com os dados originais
        graph_path (str): Caminho para salvar o arquivo .gml principal
        
    Returns:
        dict: Dicionário com os caminhos dos arquivos salvos
    """
    try:
        import os
        
        nx.write_gml(main_graph, graph_path)
        
        saved_files = {'main': graph_path}
        
        base_name = os.path.splitext(graph_path)[0]
        features_path = f"{base_name}_features.gml"
        adjacency_path = f"{base_name}_adjacency.gml"
        
        feature_matrix, feature_names = extract_feature_matrix_from_graph(main_graph)
        if feature_matrix is not None:
            features_saved = save_feature_matrix_as_gml(feature_matrix, feature_names, features_path)
            if features_saved:
                saved_files['features'] = features_path
        
        adj_matrix, edge_features = extract_adjacency_matrix_from_graph(main_graph, include_edge_features=True)
        if adj_matrix is not None:
            adjacency_saved = save_adjacency_matrix_as_gml(adj_matrix, edge_features, adjacency_path, main_graph)
            if adjacency_saved:
                saved_files['adjacency'] = adjacency_path
        
        print(f"Grafo principal salvo: {graph_path}")
        if 'features' in saved_files:
            print(f"Features salvas: {features_path}")
        if 'adjacency' in saved_files:
            print(f"Adjacência salva: {adjacency_path}")
        
        return saved_files
        
    except Exception as e:
        print(f"Erro ao salvar arquivos .gml separados: {e}")
        return {}

def save_feature_matrix_as_gml(feature_matrix, feature_names, output_path):
    """
    Salva uma matriz de features como arquivo .gml.
    
    Args:
        feature_matrix (np.ndarray): Matriz de features (n_nodes, n_features)
        feature_names (list): Lista com nomes das features
        output_path (str): Caminho para salvar o arquivo .gml
        
    Returns:
        bool: True se salvou com sucesso, False caso contrário
    """
    try:
        features_graph = nx.Graph()
        
        features_graph.graph['data_type'] = 'feature_matrix'
        features_graph.graph['shape'] = f"{feature_matrix.shape[0]},{feature_matrix.shape[1]}"
        features_graph.graph['feature_names'] = ','.join(feature_names) if feature_names else "unknown"
        
        for i, features_row in enumerate(feature_matrix):
            features_str = ','.join([f'{x:.6f}' for x in features_row])
            features_graph.add_node(i, features=features_str)
        
        nx.write_gml(features_graph, output_path)
        return True
        
    except Exception as e:
        print(f"Erro ao salvar matriz de features como .gml: {e}")
        return False

def save_adjacency_matrix_as_gml(adj_matrix, edge_features, output_path, main_graph=None):
    """
    Salva uma matriz de adjacência como arquivo .gml com informações dos nós originais.
    
    Args:
        adj_matrix (np.ndarray): Matriz de adjacência (n_nodes, n_nodes)
        edge_features (dict): Dicionário com features das arestas
        output_path (str): Caminho para salvar o arquivo .gml
        main_graph (nx.Graph, optional): Grafo original para extrair informações dos nós
        
    Returns:
        bool: True se salvou com sucesso, False caso contrário
    """
    try:
        adj_graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        
        adj_graph.graph['data_type'] = 'adjacency_matrix'
        adj_graph.graph['shape'] = f"{adj_matrix.shape[0]},{adj_matrix.shape[1]}"
        adj_graph.graph['density'] = f"{np.sum(adj_matrix) / (adj_matrix.shape[0] * adj_matrix.shape[1]):.6f}"
        adj_graph.graph['n_edges'] = int(np.sum(adj_matrix))
        
        if main_graph is not None:
            original_nodes = sorted(main_graph.nodes(data=True), key=lambda x: x[0])
            
            for i, node_id in enumerate(adj_graph.nodes()):
                if i < len(original_nodes):
                    original_node_id, original_attrs = original_nodes[i]
                    
                    node_info = {
                        'original_node_id': str(original_node_id),
                        'node_type': 'adjacency_node'
                    }
                    
                    basic_features = [
                        'start_time', 'end_time', 'fixation_duration',
                        'pupil_x', 'pupil_y', 'dispersion_x', 'dispersion_y'
                    ]
                    
                    for feature in basic_features:
                        if feature in original_attrs:
                            node_info[feature] = original_attrs[feature]
                    
                    eeg_data = original_attrs.get('eeg_data', 'NA')
                    node_info['has_eeg_data'] = str(eeg_data != 'NA' and eeg_data != '')
                    
                    adj_graph.nodes[node_id].update(node_info)
        
        if edge_features:
            for (source, target), attrs in edge_features.items():
                if adj_graph.has_edge(source, target):
                    adj_graph[source][target].update(attrs)
        
        nx.write_gml(adj_graph, output_path)
        return True
        
    except Exception as e:
        print(f"Erro ao salvar matriz de adjacência como .gml: {e}")
        return False

def load_feature_matrix_from_gml(features_path):
    """
    Carrega uma matriz de features de um arquivo .gml.
    
    Args:
        features_path (str): Caminho para o arquivo .gml de features
        
    Returns:
        tuple: (feature_matrix, feature_names) ou (None, None)
    """
    try:
        features_graph = nx.read_gml(features_path)
        
        if features_graph.graph.get('data_type') != 'feature_matrix':
            print(f"Arquivo não é uma matriz de features: {features_path}")
            return None, None
        
        feature_names_str = features_graph.graph.get('feature_names', '')
        feature_names = feature_names_str.split(',') if feature_names_str else []
        
        sorted_nodes = sorted(features_graph.nodes(data=True), key=lambda x: x[0])
        feature_rows = []
        
        for node_id, attrs in sorted_nodes:
            features_str = attrs.get('features', '')
            if features_str:
                features_row = [float(x) for x in features_str.split(',')]
                feature_rows.append(features_row)
        
        if feature_rows:
            feature_matrix = np.array(feature_rows)
            return feature_matrix, feature_names
        else:
            return None, None
            
    except Exception as e:
        print(f"Erro ao carregar matriz de features de .gml: {e}")
        return None, None

def load_adjacency_matrix_from_gml(adjacency_path):
    """
    Carrega uma matriz de adjacência de um arquivo .gml.
    
    Args:
        adjacency_path (str): Caminho para o arquivo .gml de adjacência
        
    Returns:
        tuple: (adj_matrix, edge_features) ou (None, None)
    """
    try:
        adj_graph = nx.read_gml(adjacency_path)
        
        if adj_graph.graph.get('data_type') != 'adjacency_matrix':
            print(f"Arquivo não é uma matriz de adjacência: {adjacency_path}")
            return None, None
        
        adj_matrix = nx.adjacency_matrix(adj_graph).toarray()
        
        edge_features = {}
        for source, target, attrs in adj_graph.edges(data=True):
            edge_attrs = {k: v for k, v in attrs.items() if k not in ['weight']}
            if edge_attrs:
                edge_features[(source, target)] = edge_attrs
        
        return adj_matrix, edge_features if edge_features else None
        
    except Exception as e:
        print(f"Erro ao carregar matriz de adjacência de .gml: {e}")
        return None, None