"""
Script para gerar grafos com dados EEG - versão de teste para alguns sujeitos.
"""

import sys
import os

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import pandas as pd
import networkx as nx
import numpy as np
from src.data.utils import infer_timestamps
from src.data.eeg_interpolation import process_subject_data

def generate_graphs_with_eeg(subjects=[1], sessions=[1], trials=[1, 2, 3]):
    """
    Gera grafos com dados EEG para sujeitos, sessões e trials específicos.
    
    Args:
        subjects (list): Lista de IDs de sujeitos para processamento
        sessions (list): Lista de sessões para processamento
        trials (list): Lista de trials (1-based) para processamento
    
    Returns:
        None: Salva os grafos gerados em arquivos .gml no diretório de saída
    """
    
    print("Gerando grafos com integracao de dados EEG...")
    
    eye_raw_dir = '../database/raw/Eye_raw/seed_v_eye_feature_raw_excel'
    output_base_dir = '../database/graph'
    
    for subject_id in subjects:
        subject_dir = os.path.join(output_base_dir, f'subject_{subject_id}')
        os.makedirs(subject_dir, exist_ok=True)
        
        print(f"Processando sujeito {subject_id}...")
        
        for session in sessions:
            session_dir = os.path.join(eye_raw_dir, f'Session_{session}')
            
            excel_file = None
            for file_name in os.listdir(session_dir):
                if file_name.startswith(f'{subject_id}_{session}_') and file_name.endswith('.xlsx'):
                    excel_file = file_name
                    break
            
            if not excel_file:
                print(f"  Arquivo não encontrado para sessão {session}")
                continue
            
            excel_path = os.path.join(session_dir, excel_file)
            xls = pd.ExcelFile(excel_path)
            
            print(f"  Sessão {session}: {excel_file}")
            
            for trial in trials:
                if trial-1 >= len(xls.sheet_names):
                    continue
                
                sheet_name = xls.sheet_names[trial-1]
                
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl', header=0)
                    df.columns = df.columns.str.strip()
                    
                    numeric_cols = [
                        'Fixation Duration [ms]',
                        'Average Pupil Size [px] X',
                        'Average Pupil Size [px] Y',
                        'Dispersion X',
                        'Dispersion Y',
                        'Saccade Duration [ms]',
                        'Amplitude [°]'
                    ]
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df_events = df[df['Fixation Duration [ms]'].notna()].copy()
                    
                    if df_events.empty:
                        print(f"    Trial {trial}: sem dados de fixacao validos")
                        continue
                    
                    df_events = infer_timestamps(df_events)
                    
                    print(f"    Trial {trial}: integrando dados EEG")
                    eeg_timestamps, eeg_segments, eeg_labels = process_subject_data(
                        subject_id, session, trial
                    )
                    
                    G = nx.DiGraph()
                    node_counter = 0
                    
                    for idx, row in df_events.iterrows():
                        start_time = str(row.get('start_time', 'NA'))
                        end_time = str(row.get('end_time', 'NA'))
                        fixation_duration = str(row.get('Fixation Duration [ms]', 'NA'))
                        pupil_x = str(row.get('Average Pupil Size [px] X', 'NA'))
                        pupil_y = str(row.get('Average Pupil Size [px] Y', 'NA'))
                        dispersion_x = str(row.get('Dispersion X', 'NA'))
                        dispersion_y = str(row.get('Dispersion Y', 'NA'))
                        
                        eeg_data_str = "NA"
                        if eeg_segments and node_counter < len(eeg_segments):
                            eeg_array = eeg_segments[node_counter]
                            eeg_data_str = ','.join([f'{x:.6f}' for x in eeg_array])
                        
                        node_id = idx
                        G.add_node(node_id, 
                                   start_time=start_time,
                                   end_time=end_time,
                                   fixation_duration=fixation_duration,
                                   pupil_x=pupil_x,
                                   pupil_y=pupil_y,
                                   dispersion_x=dispersion_x,
                                   dispersion_y=dispersion_y,
                                   eeg_data=eeg_data_str)
                        
                        if idx > 0:
                            prev_node = idx - 1
                            saccade_duration = str(row.get('Saccade Duration [ms]', 'NA'))
                            saccade_amplitude = str(row.get('Amplitude [°]', 'NA'))
                            G.add_edge(prev_node, node_id,
                                       saccade_duration=saccade_duration,
                                       saccade_amplitude=saccade_amplitude)
                        
                        node_counter += 1
                    
                    graph_file = os.path.join(subject_dir, f'session_{session}_trial_{trial}.gml')
                    nx.write_gml(G, graph_file)
                    
                    eeg_status = "com dados EEG" if eeg_segments else "sem EEG"
                    eeg_count = len(eeg_segments) if eeg_segments else 0
                    
                    print(f"    Trial {trial}: {G.number_of_nodes()} eventos processados, {eeg_status}")
                    
                except Exception as e:
                    print(f"    Trial {trial}: erro no processamento - {e}")
                    
        print(f"Sujeito {subject_id} processado com sucesso")

if __name__ == "__main__":
    print("Teste de geracao de grafos com dados EEG")
    print("Processando sujeito 1, sessao 1, trials 1-3")
    
    generate_graphs_with_eeg(
        subjects=[1], 
        sessions=[1], 
        trials=[1, 2, 3]
    )
    
    print("Teste concluido!")