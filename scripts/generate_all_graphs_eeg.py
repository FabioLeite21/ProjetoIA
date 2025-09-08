"""
Script para gerar TODOS os grafos com dados EEG.
"""

import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Add the project root to sys.path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import pandas as pd
import networkx as nx
import numpy as np
from src.data.utils import infer_timestamps
from src.data.eeg_interpolation import process_subject_data

def process_single_file(file_info):
    """Processa um único arquivo Excel"""
    session_dir, file_name, session_id, output_base_dir = file_info
    
    parts = file_name.split('_')
    if len(parts) < 3:
        return {'generated': 0, 'with_eeg': 0, 'errors': 0}
        
    subject_id = int(parts[0])
    
    subject_dir = os.path.join(output_base_dir, f'subject_{subject_id}')
    os.makedirs(subject_dir, exist_ok=True)
    
    excel_path = os.path.join(session_dir, file_name)
    
    trials_processed = 0
    trials_with_eeg = 0
    errors = 0
    
    try:
        # Carrega todos os dados EEG para o sujeito de uma vez
        all_eeg_data = {}
        
        with pd.ExcelFile(excel_path, engine='openpyxl') as xls:
            sheet_names = xls.sheet_names
            
            for trial_idx, sheet_name in enumerate(sheet_names):
                trial_number = trial_idx + 1
                
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                    df.columns = df.columns.str.strip()
                    
                    # Processamento otimizado de colunas numéricas
                    numeric_cols = [
                        'Fixation Duration [ms]',
                        'Average Pupil Size [px] X',
                        'Average Pupil Size [px] Y',
                        'Dispersion X',
                        'Dispersion Y',
                        'Saccade Duration [ms]',
                        'Amplitude [°]',
                        'Saccade  Count'
                    ]
                    
                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                    
                    # Extrair Saccade Count
                    saccade_count = 0
                    if 'Saccade  Count' in df.columns:
                        saccade_series = df['Saccade  Count'].dropna()
                        if not saccade_series.empty:
                            saccade_count = int(saccade_series.iloc[0])
                    
                    df_events = df[df['Fixation Duration [ms]'].notna()].copy()
                    
                    if df_events.empty:
                        continue
                    
                    df_events = infer_timestamps(df_events)
                    
                    # Cache EEG data
                    if trial_number not in all_eeg_data:
                        eeg_timestamps, eeg_segments, eeg_labels = process_subject_data(
                            subject_id, session_id, trial_number
                        )
                        all_eeg_data[trial_number] = (eeg_timestamps, eeg_segments, eeg_labels)
                    else:
                        eeg_timestamps, eeg_segments, eeg_labels = all_eeg_data[trial_number]
                    
                    # Criação otimizada do grafo
                    G = nx.DiGraph()
                    
                    # Adicionar todos os nós de uma vez
                    node_data = []
                    for node_counter, (idx, row) in enumerate(df_events.iterrows()):
                        eeg_data_str = "NA"
                        if eeg_segments and node_counter < len(eeg_segments):
                            eeg_array = eeg_segments[node_counter]
                            eeg_data_str = ','.join([f'{x:.6f}' for x in eeg_array])
                        
                        node_attrs = {
                            'start_time': str(row.get('start_time', 'NA')),
                            'end_time': str(row.get('end_time', 'NA')),
                            'fixation_duration': str(row.get('Fixation Duration [ms]', 'NA')),
                            'pupil_x': str(row.get('Average Pupil Size [px] X', 'NA')),
                            'pupil_y': str(row.get('Average Pupil Size [px] Y', 'NA')),
                            'dispersion_x': str(row.get('Dispersion X', 'NA')),
                            'dispersion_y': str(row.get('Dispersion Y', 'NA')),
                            'eeg_data': eeg_data_str
                        }
                        node_data.append((idx, node_attrs))
                    
                    G.add_nodes_from(node_data)
                    
                    # Adicionar arestas otimizado
                    if saccade_count > 0:
                        max_edges = len(df_events) - 1
                        edges_to_add = min(saccade_count, max_edges)
                        
                        edge_data = []
                        for i in range(edges_to_add):
                            if i < len(df_events) and i + 1 < len(df_events):
                                row = df_events.iloc[i]
                                edge_attrs = {
                                    'saccade_duration': str(row.get('Saccade Duration [ms]', 'NA')),
                                    'saccade_amplitude': str(row.get('Amplitude [°]', 'NA'))
                                }
                                edge_data.append((i, i + 1, edge_attrs))
                        
                        G.add_edges_from(edge_data)
                    
                    graph_file = os.path.join(subject_dir, f'session_{session_id}_trial_{trial_number}.gml')
                    nx.write_gml(G, graph_file)
                    
                    trials_processed += 1
                    if eeg_segments:
                        trials_with_eeg += 1
                        
                except Exception as e:
                    errors += 1
                    continue
                    
    except Exception as e:
        errors += 1
    
    return {
        'subject_id': subject_id,
        'generated': trials_processed, 
        'with_eeg': trials_with_eeg, 
        'errors': errors
    }

def generate_all_graphs_with_eeg():
    """Gera todos os grafos com dados EEG para todos os sujeitos e sessões."""
    
    print("Iniciando geração de grafos com dados EEG integrados...\n")
    print("Processamento paralelo - isso deve ser mais rápido agora")
    
    eye_raw_dir = '../database/raw/Eye_raw/seed_v_eye_feature_raw_excel'
    output_base_dir = '../database/graph'
    
    sessions = ['Session_1', 'Session_2', 'Session_3']
    
    # Coleta todos os arquivos para processamento
    file_tasks = []
    for session_name in sessions:
        session_number = int(session_name.split('_')[1])
        session_dir = os.path.join(eye_raw_dir, session_name)
        
        if not os.path.exists(session_dir):
            print(f"Aviso: Diretório da sessão não encontrado - {session_dir}")
            continue
        
        for file_name in sorted(os.listdir(session_dir)):
            if file_name.endswith('.xlsx') and not file_name.startswith('.'):
                file_tasks.append((session_dir, file_name, session_number, output_base_dir))
    
    print(f"Processando {len(file_tasks)} arquivos em paralelo...")
    
    total_generated = 0
    total_with_eeg = 0
    total_errors = 0
    
    # Processamento paralelo
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_file = {executor.submit(process_single_file, task): task for task in file_tasks}
        
        for future in as_completed(future_to_file):
            task = future_to_file[future]
            try:
                result = future.result()
                subject_id = result.get('subject_id', 'unknown')
                generated = result['generated']
                with_eeg = result['with_eeg']
                errors = result['errors']
                
                print(f"  Sujeito {subject_id}: {generated} grafos criados ({with_eeg} com dados EEG)")
                
                total_generated += generated
                total_with_eeg += with_eeg
                total_errors += errors
                
            except Exception as exc:
                print(f'Erro no processamento de {task[1]}: {exc}')
                total_errors += 1
    
    print(f"\nResumo do processamento:")
    print(f"{total_generated} grafos gerados no total")
    print(f"{total_with_eeg} grafos incluem dados EEG")
    print(f"{total_errors} arquivos apresentaram erros")
    if total_generated > 0:
        print(f"Taxa de sucesso na integração EEG: {total_with_eeg/total_generated*100:.1f}%")
    print("\nProcessamento concluído!")

if __name__ == "__main__":
    generate_all_graphs_with_eeg()
