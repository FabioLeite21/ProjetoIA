import numpy as np
import pickle
from scipy import interpolate
import networkx as nx

def load_processed_data(subject_id, sessions=[1, 2, 3]):
    """
    Carrega dados processados de EEG e Eye movement para um sujeito.
    
    Args:
        subject_id (int): ID do sujeito (1-16)
        sessions (list): Lista de sessões a carregar
        
    Returns:
        dict: Dados organizados por sessão e trial
    """
    data = {
        'eeg': {},
        'eye': {},
        'labels': {}
    }
    
    # Carregar dados do arquivo NPZ
    file_path = f'database/processed/EEG_DE_features/{subject_id}_123.npz'
    
    try:
        npz_data = np.load(file_path)
        eeg_data = pickle.loads(npz_data['data'].item())
        eeg_labels = pickle.loads(npz_data['label'].item())
        
        # Organizar por sessão (15 trials por sessão)
        for session in sessions:
            data['eeg'][session] = {}
            data['labels'][session] = {}
            
            for trial in range(1, 16):  # trials 1-15
                trial_idx = (session - 1) * 15 + (trial - 1)
                if trial_idx in eeg_data:
                    data['eeg'][session][trial] = eeg_data[trial_idx]
                    data['labels'][session][trial] = eeg_labels[trial_idx]
        
        print(f"Dados carregados para sujeito {subject_id}")
        return data
        
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {file_path}")
        return None
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def interpolate_to_fixation_events(processed_data, graph_timestamps):
    """
    Interpola dados processados para corresponder aos timestamps de eventos de fixação.
    
    Args:
        processed_data (np.ndarray): Dados processados (n_samples, n_features)
        graph_timestamps (list): Lista de (start_time, end_time) dos eventos de fixação
        
    Returns:
        list: Lista de arrays EEG correspondentes a cada evento de fixação
    """
    n_samples, n_features = processed_data.shape
    
    # Criar timeline dos dados processados (assumindo distribuição uniforme)
    total_duration = graph_timestamps[-1][1] - graph_timestamps[0][0]  # duração total em ms
    processed_timeline = np.linspace(0, total_duration, n_samples)
    
    eeg_segments = []
    
    for start_time, end_time in graph_timestamps:
        # Encontrar amostras que correspondem ao intervalo de fixação
        start_idx = np.searchsorted(processed_timeline, start_time)
        end_idx = np.searchsorted(processed_timeline, end_time)
        
        if start_idx == end_idx:
            # Evento muito curto - usar amostra mais próxima
            idx = min(start_idx, n_samples - 1)
            eeg_segment = processed_data[idx:idx+1, :]
        else:
            # Pegar segmento correspondente
            start_idx = max(0, start_idx)
            end_idx = min(n_samples, end_idx + 1)
            eeg_segment = processed_data[start_idx:end_idx, :]
        
        # Se o segmento for muito grande, fazer média
        if eeg_segment.shape[0] > 1:
            eeg_segment = np.mean(eeg_segment, axis=0, keepdims=True)
            
        eeg_segments.append(eeg_segment.flatten())
    
    return eeg_segments

def advanced_interpolation(processed_data, graph_timestamps, method='linear'):
    """
    Interpolação avançada usando scipy para melhor precisão temporal.
    
    Args:
        processed_data (np.ndarray): Dados processados (n_samples, n_features)
        graph_timestamps (list): Lista de (start_time, end_time) dos eventos
        method (str): Método de interpolação ('linear', 'cubic', 'nearest')
        
    Returns:
        list: Lista de features EEG interpoladas para cada evento
    """
    n_samples, n_features = processed_data.shape
    
    # Timeline dos dados processados
    total_duration = graph_timestamps[-1][1] - graph_timestamps[0][0]
    original_timeline = np.linspace(0, total_duration, n_samples)
    
    eeg_segments = []
    
    # Interpolar cada feature separadamente
    for start_time, end_time in graph_timestamps:
        # Tempo médio do evento de fixação
        event_time = (start_time + end_time) / 2.0
        
        # Interpolar todas as features para este timestamp
        interpolated_features = []
        
        for feature_idx in range(n_features):
            feature_values = processed_data[:, feature_idx]
            
            # Criar interpolador
            if method == 'linear':
                f = interpolate.interp1d(original_timeline, feature_values, 
                                       kind='linear', bounds_error=False, 
                                       fill_value='extrapolate')
            elif method == 'cubic':
                f = interpolate.interp1d(original_timeline, feature_values, 
                                       kind='cubic', bounds_error=False, 
                                       fill_value='extrapolate')
            else:  # nearest
                f = interpolate.interp1d(original_timeline, feature_values, 
                                       kind='nearest', bounds_error=False, 
                                       fill_value='extrapolate')
            
            # Interpolar para o timestamp do evento
            interpolated_value = f(event_time)
            interpolated_features.append(interpolated_value)
        
        eeg_segments.append(np.array(interpolated_features))
    
    return eeg_segments

def extract_graph_timestamps(graph_path):
    """
    Extrai timestamps dos nós de um grafo.
    
    Args:
        graph_path (str): Caminho para o arquivo .gml
        
    Returns:
        list: Lista de tuplas (start_time, end_time)
    """
    try:
        G = nx.read_gml(graph_path)
        timestamps = []
        
        # Ordenar nós por ID para manter sequência
        sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[0])
        
        for node_id, attrs in sorted_nodes:
            start_time = float(attrs['start_time'])
            end_time = float(attrs['end_time'])
            timestamps.append((start_time, end_time))
            
        return timestamps
        
    except Exception as e:
        print(f"Erro ao processar grafo {graph_path}: {e}")
        return []

def process_subject_data(subject_id, session, trial):
    """
    Processa dados de um trial específico, convertendo para timestamps dos grafos.
    
    Args:
        subject_id (int): ID do sujeito
        session (int): Número da sessão
        trial (int): Número do trial
        
    Returns:
        tuple: (timestamps, eeg_data, labels)
    """
    # Carregar dados processados
    processed_data = load_processed_data(subject_id, [session])
    
    if not processed_data or session not in processed_data['eeg']:
        return None, None, None
        
    if trial not in processed_data['eeg'][session]:
        return None, None, None
    
    # Carregar timestamps do grafo
    graph_path = f'database/graph/subject_{subject_id}/session_{session}_trial_{trial}.gml'
    timestamps = extract_graph_timestamps(graph_path)
    
    if not timestamps:
        return None, None, None
    
    # Interpolar dados EEG
    eeg_trial_data = processed_data['eeg'][session][trial]
    eeg_segments = advanced_interpolation(eeg_trial_data, timestamps, method='linear')
    
    # Labels correspondentes
    labels = processed_data['labels'][session][trial]
    
    print(f"Processado: Subject {subject_id}, Session {session}, Trial {trial}")
    print(f"  {len(timestamps)} eventos de fixação")
    print(f"  {len(eeg_segments)} segmentos EEG interpolados")
    
    return timestamps, eeg_segments, labels

if __name__ == "__main__":
    # Teste da funcionalidade
    timestamps, eeg_data, labels = process_subject_data(1, 1, 1)
    
    if timestamps and eeg_data:
        print(f"\nTeste bem-sucedido!")
        print(f"Primeiro evento: {timestamps[0]} -> EEG shape: {eeg_data[0].shape}")
        print(f"Último evento: {timestamps[-1]} -> EEG shape: {eeg_data[-1].shape}")