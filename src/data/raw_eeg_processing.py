import numpy as np
import mne
import os
from scipy.signal import butter, filtfilt
import networkx as nx
import pickle

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

def load_raw_eeg(subject_id, session):
    """
    Carrega dados EEG raw de um sujeito e sessão específicos.
    
    Args:
        subject_id (int): ID do sujeito (1-16)
        session (int): Número da sessão (1-3)
        
    Returns:
        tuple: (raw_data, sampling_freq, trial_timestamps)
    """
    raw_folder = os.path.join(project_root, 'database', 'raw', 'EEG_raw')
    
    for filename in os.listdir(raw_folder):
        if filename.startswith(f'{subject_id}_{session}_') and filename.endswith('.cnt'):
            file_path = os.path.join(raw_folder, filename)
            break
    else:
        raise FileNotFoundError(f"Arquivo EEG não encontrado para sujeito {subject_id}, sessão {session}")
    
    eeg_raw = mne.io.read_raw_cnt(file_path, preload=True, verbose=False)
    
    useless_ch = ['M1', 'M2', 'VEO', 'HEO']
    eeg_raw.drop_channels([ch for ch in useless_ch if ch in eeg_raw.ch_names])
    
    start_times = [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
    end_times = [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
    
    trial_timestamps = [(start, end) for start, end in zip(start_times, end_times)]
    
    return eeg_raw, eeg_raw.info['sfreq'], trial_timestamps

def preprocess_eeg(data_matrix, original_fs=1000, target_fs=200, lowpass=75, highpass=1):
    """
    Aplica pré-processamento exato conforme descrito:
    1. Downsampling para 200Hz
    2. Bandpass filter 1-75Hz
    
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
    
    nyquist = target_fs / 2
    low = highpass / nyquist
    high = lowpass / nyquist
    
    b, a = butter(N=4, Wn=[low, high], btype='band')
    
    filtered_data = np.zeros_like(downsampled_data)
    for ch in range(n_channels):
        filtered_data[ch, :] = filtfilt(b, a, downsampled_data[ch, :])
    
    return filtered_data

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

def extract_de_features_segment(segment, fs=200):
    """
    Extrai features DE para um segmento nas 5 bandas de frequência.
    
    Args:
        segment (np.ndarray): Segmento EEG (n_channels, n_samples)
        fs (float): Frequência de amostragem
        
    Returns:
        np.ndarray: Features DE (n_channels, 5_bands)
    """
    n_channels, n_samples = segment.shape
    
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14), 
        'beta': (14, 31),
        'gamma': (31, 50)
    }
    
    de_features = np.zeros((n_channels, 5))
    
    for ch in range(n_channels):
        channel_signal = segment[ch, :]
        
        for band_idx, (band_name, band_range) in enumerate(bands.items()):
            band_signal = extract_band_signal(channel_signal, fs, band_range)
            
            de_value = calculate_differential_entropy_gaussian(band_signal)
            
            de_features[ch, band_idx] = de_value
    
    return de_features

def extract_graph_timestamps(graph_path):
    """
    Extrai timestamps dos nós de um grafo, convertendo para ms relativos ao início.
    
    Args:
        graph_path (str): Caminho para o arquivo do grafo
        
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

def synchronize_eeg_to_fixations(processed_data, graph_timestamps, fs=200, window_size_ms=1000):
    """
    Sincroniza dados EEG processados com eventos de fixação usando janelas temporais.
    
    Args:
        processed_data (np.ndarray): Dados EEG processados (n_channels, n_samples)
        graph_timestamps (list): Timestamps dos eventos [(start_ms, end_ms), ...]
        fs (float): Frequência de amostragem (200Hz)
        window_size_ms (int): Tamanho da janela em ms para extrair DE
        
    Returns:
        list: Features DE para cada evento de fixação
    """
    synchronized_features = []
    
    window_size_samples = int(window_size_ms * fs / 1000)
    
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
        
        segment = processed_data[:, start_sample:end_sample]
        
        de_features = extract_de_features_segment(segment, fs)
        
        synchronized_features.append(de_features.flatten())
    
    return synchronized_features

def process_trial_with_raw_eeg(subject_id, session, trial):
    """
    Processa um trial usando dados EEG raw seguindo pipeline exato:
    1. Load raw 1000Hz
    2. Downsample to 200Hz  
    3. Bandpass filter 1-75Hz
    4. Extract DE features synchronized with fixations
    
    Args:
        subject_id (int): ID do sujeito
        session (int): Número da sessão  
        trial (int): Número do trial
        
    Returns:
        tuple: (graph_timestamps, synchronized_eeg_features, labels)
    """
    try:
        eeg_raw, original_fs, trial_timestamps = load_raw_eeg(subject_id, session)
        
        if trial < 1 or trial > len(trial_timestamps):
            print(f"Trial {trial} inválido")
            return None, None, None
        
        trial_start_s, trial_end_s = trial_timestamps[trial - 1]
        trial_start_sample = int(trial_start_s * original_fs)
        trial_end_sample = int(trial_end_s * original_fs)
        
        raw_data = eeg_raw.get_data()
        trial_data = raw_data[:, trial_start_sample:trial_end_sample]
        
        processed_data = preprocess_eeg(trial_data, original_fs, target_fs=200)
        
        graph_path = os.path.join(
            project_root, 'database', 'graph',
            f'subject_{subject_id}', f'session_{session}_trial_{trial}.gml'
        )
        graph_timestamps = extract_graph_timestamps(graph_path)
        
        if not graph_timestamps:
            return None, None, None
        
        synchronized_features = synchronize_eeg_to_fixations(
            processed_data, graph_timestamps, fs=200, window_size_ms=1000
        )
        
        try:
            npz_path = os.path.join(project_root, 'database', 'processed', 'EEG_DE_features', f'{subject_id}_123.npz')
            npz_data = np.load(npz_path)
            eeg_labels = pickle.loads(npz_data['label'].item())
            trial_idx = (session - 1) * 15 + (trial - 1)
            labels = eeg_labels.get(trial_idx, None)
        except:
            labels = None
        
        print(f"Processado Subject {subject_id}, Session {session}, Trial {trial} ({len(synchronized_features)} eventos)")
        
        return graph_timestamps, synchronized_features, labels
        
    except Exception as e:
        print(f"Erro: {e}")
        return None, None, None

if __name__ == "__main__":
    print("Testando pipeline completo com dados EEG raw...")
    timestamps, eeg_features, labels = process_trial_with_raw_eeg(1, 1, 1)
    
    if timestamps and eeg_features:
        print(f"Pipeline funcionando: {len(eeg_features)} eventos, {len(eeg_features[0])} features por evento")
    else:
        print("Pipeline falhou")