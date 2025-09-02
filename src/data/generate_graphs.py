import os
import pandas as pd
import networkx as nx
import numpy as np
from src.data.utils import infer_timestamps, filter_fixation_events
from src.data.eeg_interpolation import process_subject_data

EYE_RAW_DIR = 'database/raw/Eye_raw/seed_v_eye_feature_raw_excel'
GRAPHS_OUTPUT_DIR = 'database/graph'

def generate_saccadic_graphs(eye_raw_dir=EYE_RAW_DIR, output_base_dir=GRAPHS_OUTPUT_DIR):
    sessions = ['Session_1', 'Session_2', 'Session_3']
    
    for session in sessions:
        session_dir = os.path.join(eye_raw_dir, session)
        if not os.path.exists(session_dir):
            print(f"Sessão não encontrada: {session_dir}")
            continue
        
        for file_name in os.listdir(session_dir):
            if not file_name.endswith('.xlsx'):
                continue
            
            parts = file_name.split('_')
            if len(parts) < 3:
                continue
            subject_id = parts[0]
            session_id = parts[1]
            
            subject_dir = os.path.join(output_base_dir, f'subject_{subject_id}')
            os.makedirs(subject_dir, exist_ok=True)
            
            excel_path = os.path.join(session_dir, file_name)
            xls = pd.ExcelFile(excel_path)
            sheet_names = xls.sheet_names
            
            for trial_idx, sheet_name in enumerate(sheet_names):
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
                
                print(f"Colunas disponíveis em {sheet_name}: {df.columns.tolist()}")
                print(f"Tipos de dados: {df.dtypes}")
                
                df_events = df[df['Fixation Duration [ms]'].notna()].copy()
                
                if df_events.empty:
                    print(f"Sem dados de fixação válidos em {sheet_name}. Pulando...")
                    continue
                
                print(f"Primeiras linhas de df_events em {sheet_name}:\n{df_events.head()}")
                
                df_events = infer_timestamps(df_events)
                
                G = nx.DiGraph()
                
                eeg_timestamps, eeg_segments, eeg_labels = process_subject_data(
                    int(subject_id), int(session_id), trial_idx + 1
                )
                
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
                    
                    node_counter += 1
                    
                    if idx > 0:
                        prev_node = idx - 1
                        saccade_duration = str(row.get('Saccade Duration [ms]', 'NA'))
                        saccade_amplitude = str(row.get('Amplitude [°]', 'NA'))
                        G.add_edge(prev_node, node_id,
                                   saccade_duration=saccade_duration,
                                   saccade_amplitude=saccade_amplitude)
                
                graph_file = os.path.join(subject_dir, f'session_{session_id}_trial_{trial_idx+1}.gml')
                from src.data.graph_utils import save_separate_gml_files
                save_separate_gml_files(G, graph_file)
                
                eeg_status = "com dados EEG" if eeg_segments else "sem dados EEG"
                eeg_count = len(eeg_segments) if eeg_segments else 0
                print(f"Grafo gerado: {graph_file} ({eeg_status}, {eeg_count} segmentos)")

if __name__ == "__main__":
    generate_saccadic_graphs()
