#!/usr/bin/env python3
"""
Script para testar a correção do problema dos zeros nos dados EEG.
"""

import sys
import os
import numpy as np

script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

# Simular dados EEG processados (62 canais, 2000 amostras @ 200Hz = 10 segundos)
def create_simulated_eeg_data():
    """Cria dados EEG simulados com características realistas"""
    n_channels = 62
    n_samples = 2000
    fs = 200
    
    # Criar sinal EEG simulado com diferentes bandas de frequência
    time = np.arange(n_samples) / fs
    eeg_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Adicionar componentes de diferentes bandas
        delta = 0.5 * np.sin(2 * np.pi * 2 * time)  # 2 Hz (delta)
        theta = 0.3 * np.sin(2 * np.pi * 6 * time)  # 6 Hz (theta)  
        alpha = 0.4 * np.sin(2 * np.pi * 10 * time) # 10 Hz (alpha)
        beta = 0.2 * np.sin(2 * np.pi * 20 * time)  # 20 Hz (beta)
        gamma = 0.1 * np.sin(2 * np.pi * 40 * time) # 40 Hz (gamma)
        
        # Adicionar ruído
        noise = 0.1 * np.random.normal(0, 1, n_samples)
        
        eeg_data[ch, :] = delta + theta + alpha + beta + gamma + noise
    
    return eeg_data

# Simular timestamps de fixação
def create_simulated_timestamps():
    """Cria timestamps simulados de eventos de fixação"""
    timestamps = []
    current_time = 0
    
    for _ in range(10):  # 10 eventos de fixação
        duration = np.random.randint(200, 800)  # 200-800ms
        timestamps.append((current_time, current_time + duration))
        current_time += duration + np.random.randint(50, 200)  # intervalo entre fixações
    
    return timestamps

def test_eeg_processing():
    """Testa o processamento EEG com dados simulados"""
    from src.data.eeg_interpolation import synchronize_eeg_to_fixations_batch, extract_de_features_segment
    
    print("Testando correção do problema dos zeros nos dados EEG...")
    
    # Criar dados simulados
    eeg_data = create_simulated_eeg_data()
    timestamps = create_simulated_timestamps()
    
    print(f"Dados EEG simulados: {eeg_data.shape}")
    print(f"Timestamps simulados: {len(timestamps)} eventos")
    
    # Testar função corrigida
    try:
        features = synchronize_eeg_to_fixations_batch(eeg_data, timestamps)
        
        if features:
            features_array = np.array(features)
            print(f"\nFeatures extraídas: {features_array.shape}")
            
            # Análise estatística das features
            print("\nAnálise dos valores EEG:")
            print(f"Mínimo: {features_array.min():.6f}")
            print(f"Máximo: {features_array.max():.6f}")
            print(f"Média: {features_array.mean():.6f}")
            print(f"Desvio padrão: {features_array.std():.6f}")
            
            # Contar zeros (problema anterior)
            zero_count = np.sum(features_array == 0.0)
            total_values = features_array.size
            zero_percentage = (zero_count / total_values) * 100
            
            print(f"\nAnálise de zeros:")
            print(f"Valores zero: {zero_count}/{total_values} ({zero_percentage:.2f}%)")
            
            # Testar um segmento individual para comparação
            print(f"\nTeste de um segmento individual:")
            segment = eeg_data[:, 100:300]  # 200 amostras
            de_features = extract_de_features_segment(segment)
            print(f"Features DE segment individual: {de_features.shape}")
            print(f"Valores segment: min={de_features.min():.6f}, max={de_features.max():.6f}")
            
            # Verificar se a correção funcionou
            if zero_percentage < 20:  # Se menos de 20% são zeros, provavelmente está funcionando
                print("\n✅ SUCESSO: Correção aplicada com sucesso! Poucos zeros detectados.")
            else:
                print("\n❌ PROBLEMA: Ainda há muitos zeros nos dados.")
                
        else:
            print("❌ ERRO: Nenhuma feature foi extraída")
            
    except Exception as e:
        print(f"❌ ERRO no processamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_eeg_processing()