import os
import pandas as pd
import networkx as nx
import numpy as np
from src.data.utils import infer_timestamps, filter_fixation_events
from src.data.eeg_interpolation import process_subject_data

EYE_RAW_DIR = 'database/raw/Eye_raw/seed_v_eye_feature_raw_excel'
GRAPHS_OUTPUT_DIR = 'database/graph'

def generate_saccadic_graphs(eye_raw_dir=EYE_RAW_DIR, output_base_dir=GRAPHS_OUTPUT_DIR):
    """Generate saccadic graphs from eye tracking data across multiple sessions.
    
    Processes eye tracking data from Excel files to create directed graphs where nodes
    represent fixation events and edges represent saccadic movements. Each graph
    includes EEG data interpolation for nodes and is saved in GML format.
    
    Args:
        eye_raw_dir (str): Directory path containing raw eye tracking data organized
            by sessions. Defaults to EYE_RAW_DIR.
        output_base_dir (str): Base directory path where generated graphs will be saved.
            Defaults to GRAPHS_OUTPUT_DIR.
    
    Returns:
        None: Saves graphs as GML files in subject-specific directories.
    """
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
                    'Amplitude [°]',
                    'Saccade  Count'
                ]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                
                saccade_count = 0
                if 'Saccade  Count' in df.columns:
                    saccade_series = df['Saccade  Count'].dropna()
                    if not saccade_series.empty:
                        saccade_count = int(saccade_series.iloc[0])
                        print(f"Saccade Count extraido: {saccade_count}")
                
                df_events = df[df['Fixation Duration [ms]'].notna()].copy()
                
                if df_events.empty:
                    print(f"Sem dados de fixacao validos em {sheet_name}")
                    continue
                
                
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
                
                if saccade_count > 0:
                    max_edges = len(df_events) - 1
                    edges_to_add = min(saccade_count, max_edges)
                    for i in range(edges_to_add):
                        prev_node = i
                        next_node = i + 1
                        if prev_node < len(df_events) and next_node < len(df_events):
                            row = df_events.iloc[i]
                            saccade_duration = str(row.get('Saccade Duration [ms]', 'NA'))
                            saccade_amplitude = str(row.get('Amplitude [°]', 'NA'))
                            G.add_edge(prev_node, next_node,
                                       saccade_duration=saccade_duration,
                                       saccade_amplitude=saccade_amplitude)
                    print(f"Adicionadas {edges_to_add} arestas")
                
                graph_file = os.path.join(subject_dir, f'session_{session_id}_trial_{trial_idx+1}.gml')
                nx.write_gml(G, graph_file)
                
                eeg_status = "com dados EEG" if eeg_segments else "sem dados EEG"
                eeg_count = len(eeg_segments) if eeg_segments else 0
                print(f"Grafo gerado: {graph_file} ({eeg_status})")

if __name__ == "__main__":
    generate_saccadic_graphs()