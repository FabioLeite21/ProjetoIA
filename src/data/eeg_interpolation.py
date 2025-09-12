import numpy as np
import mne
import os
import warnings
import time
import gc
from collections import OrderedDict
from scipy.signal import butter, filtfilt
import networkx as nx
import pickle
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore', message='Could not parse meas date from the header')
mne.set_log_level('ERROR')

SESSION_CACHE = OrderedDict()
MAX_CACHE_SIZE = 8
CACHE_STATS = {'hits': 0, 'misses': 0, 'evictions': 0, 'memory_saved_hours': 0}
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

def load_trial_timestamps(session):
    """
    Carrega timestamps corretos para uma sessão específica do arquivo oficial.
    
    Args:
        session (int): Número da sessão (1-3)
        
    Returns:
        list: Lista de tuplas (start_time, end_time) em segundos
    """
    timestamp_file = os.path.join(project_root, 'database', 'trial_start_end_timestamp.txt')
    
    try:
        with open(timestamp_file, 'r') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        
        session_key = f"Session {session}:"
        start_line = None
        end_line = None
        
        for i, line in enumerate(lines):
            if line.strip() == session_key:
                if i + 1 < len(lines):
                    start_line = lines[i + 1]
                if i + 2 < len(lines):
                    end_line = lines[i + 2]
                break
        
        if not start_line or not end_line:
            raise ValueError(f"Timestamps não encontrados para Session {session}")
        
        import re
        
        start_match = re.search(r'start_second:\s*\[([\d,\s]+)\]', start_line)
        end_match = re.search(r'end_second:\s*\[([\d,\s]+)\]', end_line)
        
        if not start_match or not end_match:
            raise ValueError(f"Formato inválido nos timestamps da Session {session}")
        
        start_times = [int(x.strip()) for x in start_match.group(1).split(',')]
        end_times = [int(x.strip()) for x in end_match.group(1).split(',')]
        
        if len(start_times) != len(end_times):
            raise ValueError(f"Número de start/end timestamps não bate para Session {session}")
        
        trial_timestamps = [(start, end) for start, end in zip(start_times, end_times)]
        print(f"Timestamps carregados para Session {session}: {len(trial_timestamps)} trials")
        
        return trial_timestamps
        
    except Exception as e:
        print(f"Erro ao carregar timestamps: {e}")
        print("Usando fallback: divisao automatica em 15 trials")
        return generate_fallback_timestamps()

def generate_fallback_timestamps():
    """
    Gera timestamps fallback dividindo uniformemente em 15 trials.
    
    Returns:
        list: Lista de tuplas (start_time, end_time) em segundos
    """
    start_times = [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
    end_times = [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
    return [(start, end) for start, end in zip(start_times, end_times)]

def validate_timestamps(timestamps, total_duration, subject_id, session):
    """
    Valida timestamps contra duração real do arquivo e ajusta/remove inválidos.
    
    Args:
        timestamps (list): Lista de (start, end) em segundos
        total_duration (float): Duração total do arquivo em segundos
        subject_id (int): Para logging
        session (int): Para logging
        
    Returns:
        list: Timestamps validados e ajustados
    """
    valid_timestamps = []
    invalid_count = 0
    
    for i, (start, end) in enumerate(timestamps):
        if end <= total_duration:
            valid_timestamps.append((start, end))
        elif start < total_duration:
            adjusted_end = total_duration - 1
            if adjusted_end > start:
                valid_timestamps.append((start, adjusted_end))
            else:
                invalid_count += 1
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"Aviso: {invalid_count} trials removidos/ajustados para Subject {subject_id}, Session {session}")
    
    return valid_timestamps

def load_raw_eeg_robust(subject_id, session):
    """
    Carrega dados EEG raw com fallbacks robustos para diferentes formatos.
    
    Args:
        subject_id (int): ID do sujeito (1-16)
        session (int): Número da sessão (1-3)
        
    Returns:
        tuple: (raw_data, sampling_freq, trial_timestamps, load_method)
    """
    raw_folder = os.path.join(project_root, 'database', 'raw', 'EEG_raw')
    
    file_path = None
    for filename in os.listdir(raw_folder):
        if filename.startswith(f'{subject_id}_{session}_') and filename.endswith('.cnt'):
            file_path = os.path.join(raw_folder, filename)
            break
    
    if not file_path:
        raise FileNotFoundError(f"Arquivo EEG não encontrado para sujeito {subject_id}, sessão {session}")
    
    trial_timestamps = load_trial_timestamps(session)
    
    try:
        print(f"Carregando raw (Neuroscan): Subject {subject_id}, Session {session}")
        eeg_raw = mne.io.read_raw_cnt(file_path, preload=True, verbose=False)
        
        useless_ch = ['M1', 'M2', 'VEO', 'HEO']
        eeg_raw.drop_channels([ch for ch in useless_ch if ch in eeg_raw.ch_names])
        
        total_duration = eeg_raw.n_times / eeg_raw.info['sfreq']
        validated_timestamps = validate_timestamps(trial_timestamps, total_duration, subject_id, session)
        
        return eeg_raw, eeg_raw.info['sfreq'], validated_timestamps, 'neuroscan_cnt'
        
    except (OverflowError, ValueError, RuntimeError) as e:
        print(f"Neuroscan falhou: {str(e)[:50]}")
        
        try:
            print(f"Tentando ANT Neuro: Subject {subject_id}, Session {session}")
            if subject_id in [7, 12]:
                print(f"Skip: Subject {subject_id} conhecido por causar crash no ANT Neuro")
                raise RuntimeError(f"Subject {subject_id} requires fallback")
            
            eeg_raw = mne.io.read_raw_ant(file_path, preload=True, verbose=False)
            
            useless_ch = ['M1', 'M2', 'VEO', 'HEO']
            existing_useless = [ch for ch in useless_ch if ch in eeg_raw.ch_names]
            if existing_useless:
                eeg_raw.drop_channels(existing_useless)
            
            total_duration = eeg_raw.n_times / eeg_raw.info['sfreq']
            validated_timestamps = validate_timestamps(trial_timestamps, total_duration, subject_id, session)
            
            return eeg_raw, eeg_raw.info['sfreq'], validated_timestamps, 'ant_cnt'
            
        except Exception as e2:
            print(f"ANT Neuro falhou: {str(e2)[:50]}")
            
            try:
                print(f"Usando fallback (preprocessed): Subject {subject_id}, Session {session}")
                return load_preprocessed_fallback(subject_id, session, trial_timestamps)
                
            except Exception as e3:
                print(f"Fallback falhou: {str(e3)[:50]}")
                raise RuntimeError(f"Todas as tentativas falharam para Subject {subject_id}, Session {session}")

def load_preprocessed_fallback_direct(subject_id, session, trial):
    """
    Fallback BACKUP que usa dados já processados da pasta database/backup/ quando carregamento raw falha.
    
    Args:
        subject_id (int): ID do sujeito
        session (int): Número da sessão  
        trial (int): Número do trial
        
    Returns:
        tuple: (graph_timestamps, synchronized_eeg_features, labels) ou (None, None, None)
    """
    try:
        print(f"Fallback backup: Carregando dados Subject {subject_id}, Session {session}, Trial {trial}")
        
        backup_graph_path = os.path.join(
            project_root, 'database', 'backup',
            f'subject_{subject_id}', f'session_{session}_trial_{trial}.gml'
        )
        
        if not os.path.exists(backup_graph_path):
            print(f"Fallback backup: Arquivo nao encontrado: {backup_graph_path}")
            return None, None, None
        
        G = nx.read_gml(backup_graph_path)
        
        if G.number_of_nodes() == 0:
            print(f"Fallback backup: Grafo vazio")
            return None, None, None
        
        graph_timestamps = []
        synchronized_features = []
        
        sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[0])
        
        for node_id, attrs in sorted_nodes:
            start_time = float(attrs.get('start_time', 0))
            end_time = float(attrs.get('end_time', 0))
            graph_timestamps.append((start_time, end_time))
            
            eeg_data_str = attrs.get('eeg_data', '')
            if eeg_data_str and eeg_data_str != 'NA':
                try:
                    eeg_values = np.array([float(x) for x in eeg_data_str.split(',')])
                    if len(eeg_values) == 310:
                        synchronized_features.append(eeg_values)
                    else:
                        print(f"Fallback backup: EEG data com dimensao incorreta: {len(eeg_values)} (esperado 310)")
                        return None, None, None
                except ValueError as e:
                    print(f"Fallback backup: Erro ao converter eeg_data: {e}")
                    return None, None, None
            else:
                print(f"Fallback backup: EEG data ausente no no {node_id}")
                return None, None, None
        
        if not synchronized_features:
            print(f"Fallback backup: Nenhuma feature EEG extraida")
            return None, None, None
        
        try:
            npz_path = os.path.join(project_root, 'database', 'processed', 'EEG_DE_features', f'{subject_id}_123.npz')
            npz_data = np.load(npz_path)
            eeg_labels = pickle.loads(npz_data['label'].item())
            trial_idx = (session - 1) * 15 + (trial - 1)
            labels = eeg_labels.get(trial_idx, None)
        except:
            labels = None
        
        print(f"Fallback backup: {len(graph_timestamps)} eventos processados")
        
        return graph_timestamps, synchronized_features, labels
        
    except Exception as e:
        print(f"Fallback backup erro: {e}")
        return None, None, None

def load_preprocessed_fallback(subject_id, session, trial_timestamps):
    """
    Fallback antigo mantido para compatibilidade.
    
    Args:
        subject_id (int): ID do sujeito
        session (int): Número da sessão
        trial_timestamps (list): Lista de timestamps dos trials
        
    Raises:
        NotImplementedError: Esta função não está mais implementada
    """
    raise NotImplementedError("Use load_preprocessed_fallback_direct() em vez disso")

def get_cached_session(subject_id, session):
    """
    Obtém sessão do cache ou carrega nova com LRU eviction.
    
    Args:
        subject_id (int): ID do sujeito
        session (int): Número da sessão
        
    Returns:
        tuple: Dados da sessão (raw_data, sampling_freq, trial_timestamps, load_method)
    """
    global SESSION_CACHE, CACHE_STATS
    
    cache_key = (subject_id, session)
    
    if cache_key in SESSION_CACHE:
        CACHE_STATS['hits'] += 1
        SESSION_CACHE.move_to_end(cache_key)
        return SESSION_CACHE[cache_key]
    
    CACHE_STATS['misses'] += 1
    
    if len(SESSION_CACHE) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(SESSION_CACHE))
        print(f"Cache eviction: {oldest_key}")
        del SESSION_CACHE[oldest_key]
        CACHE_STATS['evictions'] += 1
        gc.collect()
    
    start_time = time.time()
    session_data = load_raw_eeg_robust(subject_id, session)
    load_time = time.time() - start_time
    print(f"  Loaded in {load_time:.1f}s ({session_data[3]})")
    
    SESSION_CACHE[cache_key] = session_data
    
    return session_data

def load_raw_eeg(subject_id, session):
    """
    Interface pública que usa cache inteligente para carregar dados EEG raw.
    
    Args:
        subject_id (int): ID do sujeito
        session (int): Número da sessão
        
    Returns:
        tuple: (raw_data, sampling_freq, trial_timestamps)
    """
    return get_cached_session(subject_id, session)[:3]

def safe_filtfilt(b, a, data, min_length=30):
    """
    Aplica filtfilt com padding inteligente para sinais curtos.
    
    Args:
        b (np.ndarray): Coeficientes do numerador do filtro
        a (np.ndarray): Coeficientes do denominador do filtro
        data (np.ndarray): Sinal 1D
        min_length (int): Comprimento mínimo necessário
        
    Returns:
        np.ndarray: Sinal filtrado
    """
    if len(data) >= min_length:
        return filtfilt(b, a, data)
    
    pad_needed = min_length - len(data) + 10
    pad_left = pad_needed // 2
    pad_right = pad_needed - pad_left
    
    padded_data = np.pad(data, (pad_left, pad_right), mode='edge')
    
    filtered_padded = filtfilt(b, a, padded_data)
    
    return filtered_padded[pad_left:pad_left + len(data)]

def preprocess_eeg(data_matrix, original_fs=1000, target_fs=200, lowpass=75, highpass=1):
    """
    Aplica pré-processamento robusto com proteção contra trials curtos:
    1. Downsampling para 200Hz
    2. Bandpass filter 1-75Hz (com padding inteligente se necessário)
    
    Args:
        data_matrix (np.ndarray): Dados raw (n_channels, n_samples)
        original_fs (float): Frequência de amostragem original
        target_fs (float): Frequência de amostragem alvo (200Hz)
        lowpass (float): Frequência de corte superior
        highpass (float): Frequência de corte inferior
        
    Returns:
        np.ndarray: Dados processados (n_channels, n_samples_downsampled)
    """
    n_channels, n_samples = data_matrix.shape
    
    downsample_factor = int(original_fs / target_fs)
    downsampled_data = data_matrix[:, ::downsample_factor]
    
    downsampled_samples = downsampled_data.shape[1]
    
    min_samples_needed = 30
    
    original_duration_samples = n_samples
    original_duration_ms = (original_duration_samples / original_fs) * 1000
    downsampled_duration_ms = (downsampled_samples / target_fs) * 1000
    
    if downsampled_samples < min_samples_needed:
        print(f"      AVISO: Trial curto (original: {original_duration_ms:.0f}ms -> {downsampled_duration_ms:.0f}ms, {downsampled_samples} samples) - usando padding")
    elif downsampled_samples < 100:
        print(f"      INFO: Trial pequeno ({downsampled_duration_ms:.0f}ms, {downsampled_samples} samples)")
    
    nyquist = target_fs / 2
    low = highpass / nyquist
    high = lowpass / nyquist
    
    try:
        b, a = butter(N=4, Wn=[low, high], btype='band')
        
        filtered_data = np.zeros_like(downsampled_data)
        for ch in range(n_channels):
            filtered_data[ch, :] = safe_filtfilt(b, a, downsampled_data[ch, :], min_samples_needed)
        
        return filtered_data
        
    except Exception as e:
        print(f"      ERRO no filtro Butterworth N=4: {e}")
        
        try:
            print(f"      Tentando filtro N=2...")
            b2, a2 = butter(N=2, Wn=[low, high], btype='band')
            
            filtered_data = np.zeros_like(downsampled_data)
            for ch in range(n_channels):
                filtered_data[ch, :] = safe_filtfilt(b2, a2, downsampled_data[ch, :], 15)
            
            print(f"      Sucesso com filtro N=2")
            return filtered_data
            
        except Exception as e2:
            print(f"      ERRO no filtro N=2: {e2}")
            
            print(f"      FALLBACK: Retornando dados sem filtro (apenas downsampled)")
            return downsampled_data

def calculate_differential_entropy_gaussian(signal):
    """
    Calcula Differential Entropy assumindo distribuição Gaussiana.
    Fórmula: DE = 1/2 * ln(2πeσ²)
    
    Args:
        signal (np.ndarray): Sinal 1D
        
    Returns:
        float: Valor DE
    """
    if len(signal) == 0:
        return 0.0
    
    variance = np.var(signal)
    
    if variance <= 0:
        return 0.0
    
    de = 0.5 * np.log(2 * np.pi * np.e * variance)
    
    return de

def extract_band_signal(signal, fs, band_range):
    """
    Extrai sinal de uma banda de frequência específica usando FFT.
    
    Args:
        signal (np.ndarray): Sinal temporal
        fs (float): Frequência de amostragem
        band_range (tuple): (freq_min, freq_max)
        
    Returns:
        np.ndarray: Sinal filtrado na banda
    """
    freq_min, freq_max = band_range
    
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
    
    fft_filtered = fft_signal.copy()
    fft_filtered[~mask] = 0
    
    filtered_signal = np.real(np.fft.ifft(fft_filtered))
    
    return filtered_signal

def create_band_masks(n_samples, fs):
    """
    Cria máscaras pré-computadas para todas as bandas de frequência.
    
    Args:
        n_samples (int): Número de amostras do sinal
        fs (float): Frequência de amostragem
        
    Returns:
        dict: Dicionário com máscaras para cada banda de frequência
    """
    freqs = np.fft.fftfreq(n_samples, 1/fs)
    
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14), 
        'beta': (14, 31),
        'gamma': (31, 50)
    }
    
    masks = {}
    for band_name, (freq_min, freq_max) in bands.items():
        mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        masks[band_name] = mask
    
    return masks

def batch_gaussian_de(signals):
    """
    Calcula DE Gaussiana para batch de sinais simultaneamente.
    
    Args:
        signals (np.ndarray): (n_events, n_channels, n_samples) ou (n_channels, n_samples)
        
    Returns:
        np.ndarray: DE values
    """
    if signals.ndim == 3:
        variances = np.var(signals, axis=2)
    else:
        variances = np.var(signals, axis=1)
    
    variances = np.maximum(variances, 1e-10)
    
    de_values = 0.5 * np.log(2 * np.pi * np.e * variances)
    
    return de_values

def extract_de_features_batch(all_segments, fs=200):
    """
    Extrai features DE para múltiplos segmentos simultaneamente usando abordagem robusta.
    
    Args:
        all_segments (np.ndarray): (n_events, n_channels, n_samples)
        fs (float): Frequência de amostragem
        
    Returns:
        np.ndarray: Features DE (n_events, n_channels, 5_bands)
    """
    n_events, n_channels, n_samples = all_segments.shape
    
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14), 
        'beta': (14, 31),
        'gamma': (31, 50)
    }
    
    de_features = np.zeros((n_events, n_channels, 5))
    
    for event_idx in range(n_events):
        segment = all_segments[event_idx]  # (n_channels, n_samples)
        
        for ch in range(n_channels):
            channel_signal = segment[ch, :]
            
            for band_idx, (band_name, (freq_min, freq_max)) in enumerate(bands.items()):
                # Usar filtragem FFT robusta
                fft_signal = np.fft.fft(channel_signal)
                freqs = np.fft.fftfreq(len(channel_signal), 1/fs)
                
                # Criar máscara para a banda de frequência
                mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
                
                # Aplicar filtro
                fft_filtered = fft_signal.copy()
                fft_filtered[~mask] = 0
                
                # Converter de volta para domínio temporal
                band_signal = np.real(np.fft.ifft(fft_filtered))
                
                # Calcular DE Gaussiana
                variance = np.var(band_signal)
                if variance <= 0:
                    de_value = 0.0
                else:
                    de_value = 0.5 * np.log(2 * np.pi * np.e * variance)
                
                de_features[event_idx, ch, band_idx] = de_value
    
    return de_features

def extract_de_features_segment(segment, fs=200):
    """
    Interface compatível que usa processamento batch para um único segmento.
    
    Args:
        segment (np.ndarray): Segmento EEG (n_channels, n_samples)
        fs (float): Frequência de amostragem
        
    Returns:
        np.ndarray: Features DE (n_channels, 5_bands)
    """
    batch_segment = segment[np.newaxis, :, :]
    batch_features = extract_de_features_batch(batch_segment, fs)
    return batch_features[0]

def extract_graph_timestamps(graph_path):
    """
    Extrai timestamps dos nós de um grafo, convertendo para ms relativos ao início.
    
    Args:
        graph_path (str): Caminho para o arquivo .gml do grafo
        
    Returns:
        list: Lista de tuplas (start_time, end_time) em ms relativos
    """
    try:
        G = nx.read_gml(graph_path)
        
        timestamps = []
        sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[0])
        
        if sorted_nodes:
            first_start_time = float(sorted_nodes[0][1]['start_time'])
        else:
            return []
        
        for node_id, attrs in sorted_nodes:
            start_time = float(attrs['start_time']) - first_start_time
            end_time = float(attrs['end_time']) - first_start_time
            timestamps.append((start_time, end_time))
            
        return timestamps
        
    except Exception as e:
        print(f"Erro ao processar grafo {graph_path}: {e}")
        return []

def synchronize_eeg_to_fixations_batch(processed_data, graph_timestamps, fs=200, window_size_ms=1000):
    """
    Sincroniza dados EEG com eventos de fixação usando processamento individual (sem padding zeros).
    Reverte para lógica original que funcionava, processando cada segmento individualmente.
    
    Args:
        processed_data (np.ndarray): Dados EEG processados (n_channels, n_samples)
        graph_timestamps (list): Timestamps dos eventos [(start_ms, end_ms), ...]
        fs (float): Frequência de amostragem (200Hz)
        window_size_ms (int): Tamanho da janela em ms para extrair DE
        
    Returns:
        list: Features DE normalizadas para cada evento de fixação
    """
    if not graph_timestamps:
        return []
    
    synchronized_features = []
    window_size_samples = int(window_size_ms * fs / 1000)
    
    print(f"    Processamento sem padding: {len(graph_timestamps)} eventos")
    
    # Processar cada evento individualmente (como na versão original)
    for fix_start_ms, fix_end_ms in graph_timestamps:
        fix_center_ms = (fix_start_ms + fix_end_ms) / 2
        fix_center_sample = int(fix_center_ms * fs / 1000)
        
        start_sample = max(0, fix_center_sample - window_size_samples // 2)
        end_sample = min(processed_data.shape[1], start_sample + window_size_samples)
        
        if end_sample - start_sample < window_size_samples:
            if start_sample == 0:
                end_sample = min(window_size_samples, processed_data.shape[1])
            else:
                start_sample = max(0, end_sample - window_size_samples)
        
        # CRÍTICO: Não adicionar padding - usar segmento natural
        segment = processed_data[:, start_sample:end_sample]
        
        # Processar com tamanho natural (sem forçar tamanho fixo)
        de_features = extract_de_features_segment(segment, fs)
        synchronized_features.append(de_features.flatten())
    
    # Aplicar normalização MinMax apenas nos valores reais (sem zeros artificiais)
    if synchronized_features:
        features_matrix = np.array(synchronized_features)
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(features_matrix)
        return [normalized_features[i] for i in range(len(synchronized_features))]
    
    return synchronized_features

def synchronize_eeg_to_fixations(processed_data, graph_timestamps, fs=200, window_size_ms=1000):
    """
    Interface compatível que usa processamento batch por padrão.
    
    Args:
        processed_data (np.ndarray): Dados EEG processados (n_channels, n_samples)
        graph_timestamps (list): Timestamps dos eventos [(start_ms, end_ms), ...]
        fs (float): Frequência de amostragem (200Hz)
        window_size_ms (int): Tamanho da janela em ms para extrair DE
        
    Returns:
        list: Features DE normalizadas para cada evento de fixação
    """
    return synchronize_eeg_to_fixations_batch(processed_data, graph_timestamps, fs, window_size_ms)

def print_cache_stats():
    """
    Mostra estatísticas do cache de memória.
    """
    global CACHE_STATS
    
    total_requests = CACHE_STATS['hits'] + CACHE_STATS['misses']
    if total_requests > 0:
        hit_rate = CACHE_STATS['hits'] / total_requests * 100
        print(f"Cache Stats: {CACHE_STATS['hits']} hits, {CACHE_STATS['misses']} misses ({hit_rate:.1f}% hit rate)")
        print(f"   Memory: {len(SESSION_CACHE)}/{MAX_CACHE_SIZE} sessoes cached (~{len(SESSION_CACHE)*3:.1f}GB)")
        if CACHE_STATS['hits'] > 0:
            time_saved = CACHE_STATS['hits'] * 30
            print(f"   Tempo economizado: ~{time_saved//60:.0f}min {time_saved%60:.0f}s")

def process_subject_data(subject_id, session, trial):
    """
    Processa um trial usando dados EEG raw com cache inteligente e batch processing:
    1. Smart skip se já processado
    2. Load raw com cache (1000Hz)
    3. Downsample to 200Hz  
    4. Bandpass filter 1-75Hz
    5. Extract DE features com ULTRA-BATCH
    6. Normalização MinMax
    
    Args:
        subject_id (int): ID do sujeito
        session (int): Número da sessão  
        trial (int): Número do trial
        
    Returns:
        tuple: (graph_timestamps, synchronized_eeg_features, labels)
    """
    try:
        start_time = time.time()
        
        try:
            eeg_raw, original_fs, trial_timestamps = load_raw_eeg(subject_id, session)
        except Exception as e:
            print(f"FALLBACK BACKUP: Raw loading falhou para Subject {subject_id}, Session {session} - {e}")
            return load_preprocessed_fallback_direct(subject_id, session, trial)
        
        if trial < 1 or trial > len(trial_timestamps):
            print(f"Trial {trial} inválido")
            return None, None, None
        
        trial_start_s, trial_end_s = trial_timestamps[trial - 1]
        trial_start_sample = int(trial_start_s * original_fs)
        trial_end_sample = int(trial_end_s * original_fs)
        
        raw_data = eeg_raw.get_data()
        trial_data = raw_data[:, trial_start_sample:trial_end_sample]
        
        preprocess_start = time.time()
        try:
            processed_data = preprocess_eeg(trial_data, original_fs, target_fs=200)
        except Exception as e:
            print(f"FALLBACK BACKUP: Preprocessing falhou para Subject {subject_id}, Session {session}, Trial {trial} - {e}")
            return load_preprocessed_fallback_direct(subject_id, session, trial)
        preprocess_time = time.time() - preprocess_start
        
        # For graph generation, we return raw processed EEG data without synchronization
        # The synchronization will happen in the graph generation script with eye-tracking timestamps
        sync_start = time.time()
        
        # Extract DE features for the entire trial in segments
        # Use a fixed window size to create consistent feature segments
        window_size_ms = 1000
        window_size_samples = int(window_size_ms * 200 / 1000)  # 200 samples for 1000ms at 200Hz
        
        n_channels, n_samples = processed_data.shape
        synchronized_features = []
        
        # Create overlapping windows across the entire trial
        step_size = window_size_samples // 2  # 50% overlap
        for start_sample in range(0, n_samples - window_size_samples + 1, step_size):
            end_sample = start_sample + window_size_samples
            segment = processed_data[:, start_sample:end_sample]
            
            de_features = extract_de_features_segment(segment, fs=200)
            synchronized_features.append(de_features.flatten())
        
        # Apply normalization
        if synchronized_features:
            features_matrix = np.array(synchronized_features)
            scaler = MinMaxScaler()
            normalized_features = scaler.fit_transform(features_matrix)
            synchronized_features = [normalized_features[i] for i in range(len(synchronized_features))]
        
        sync_time = time.time() - sync_start
        
        try:
            npz_path = os.path.join(project_root, 'database', 'processed', 'EEG_DE_features', f'{subject_id}_123.npz')
            npz_data = np.load(npz_path)
            eeg_labels = pickle.loads(npz_data['label'].item())
            trial_idx = (session - 1) * 15 + (trial - 1)
            labels = eeg_labels.get(trial_idx, None)
        except:
            labels = None
        
        total_time = time.time() - start_time
        
        print(f"Processado: Subject {subject_id}, Session {session}, Trial {trial} ({len(synchronized_features)} eventos)")
        
        
        return None, synchronized_features, labels
        
    except Exception as e:
        print(f"Erro Subject {subject_id}, Session {session}, Trial {trial}: {str(e)[:60]}")
        return None, None, None

if __name__ == "__main__":
    print("Testando pipeline completo com dados EEG raw...")
    timestamps, eeg_features, labels = process_subject_data(1, 1, 1)
    
    if timestamps and eeg_features:
        print(f"Pipeline funcionando: {len(eeg_features)} eventos, {len(eeg_features[0])} features por evento")
    else:
        print("Pipeline falhou")