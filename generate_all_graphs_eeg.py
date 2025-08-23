"""
Script para gerar TODOS os grafos com dados EEG.
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
from src.data.utils import infer_timestamps
from src.data.eeg_interpolation import process_subject_data

def generate_all_graphs_with_eeg():
    """Gera todos os grafos com dados EEG para todos os sujeitos e sessões."""
    
    print("Iniciando geração de grafos com dados EEG integrados...\n")
    print("Processamento completo - isso pode levar alguns minutos")
    
    eye_raw_dir = 'database/raw/Eye_raw/seed_v_eye_feature_raw_excel'
    output_base_dir = 'database/graph'
    
    total_generated = 0
    total_with_eeg = 0
    total_errors = 0
    
    # Processar todas as sessões
    sessions = ['Session_1', 'Session_2', 'Session_3']
    
    for session_name in sessions:
        session_number = int(session_name.split('_')[1])
        session_dir = os.path.join(eye_raw_dir, session_name)
        
        if not os.path.exists(session_dir):
            print(f"Aviso: Diretório da sessão não encontrado - {session_dir}")
            continue
        
        print(f"\nProcessando {session_name.replace('_', ' ')}...")
        
        # Processar todos os arquivos da sessão
        for file_name in sorted(os.listdir(session_dir)):
            if not file_name.endswith('.xlsx') or file_name.startswith('.'):
                continue
            
            # Extrair informações do arquivo
            parts = file_name.split('_')
            if len(parts) < 3:
                continue
                
            subject_id = int(parts[0])
            session_id = int(parts[1])
            
            print(f"  Sujeito {subject_id}: ", end="", flush=True)
            
            # Criar diretório do sujeito
            subject_dir = os.path.join(output_base_dir, f'subject_{subject_id}')
            os.makedirs(subject_dir, exist_ok=True)
            
            excel_path = os.path.join(session_dir, file_name)
            
            try:
                xls = pd.ExcelFile(excel_path)
                sheet_names = xls.sheet_names
                
                trials_processed = 0
                trials_with_eeg = 0
                
                for trial_idx, sheet_name in enumerate(sheet_names):
                    trial_number = trial_idx + 1
                    
                    try:
                        # Ler dados eye tracking
                        df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl', header=0)
                        df.columns = df.columns.str.strip()
                        
                        # Converter colunas para numérico
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
                        
                        # Filtrar eventos de fixação
                        df_events = df[df['Fixation Duration [ms]'].notna()].copy()
                        
                        if df_events.empty:
                            continue
                        
                        # Inferir timestamps
                        df_events = infer_timestamps(df_events)
                        
                        # Carregar dados EEG
                        eeg_timestamps, eeg_segments, eeg_labels = process_subject_data(
                            subject_id, session_id, trial_number
                        )
                        
                        # Criar grafo
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
                            
                            # Obter dados EEG
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
                        
                        # Salvar grafo
                        graph_file = os.path.join(subject_dir, f'session_{session_id}_trial_{trial_number}.gml')
                        nx.write_gml(G, graph_file)
                        
                        trials_processed += 1
                        total_generated += 1
                        
                        if eeg_segments:
                            trials_with_eeg += 1
                            total_with_eeg += 1
                        
                    except Exception as e:
                        total_errors += 1
                        continue
                
                # Status do sujeito
                print(f"{trials_processed} grafos criados ({trials_with_eeg} com dados EEG)")
                
            except Exception as e:
                print(f"erro no processamento - {str(e)[:50]}...")
                total_errors += 1
    
    print(f"\nResumo do processamento:")
    print(f"• {total_generated} grafos gerados no total")
    print(f"• {total_with_eeg} grafos incluem dados EEG")
    print(f"• {total_errors} arquivos apresentaram erros")
    if total_generated > 0:
        print(f"• Taxa de sucesso na integração EEG: {total_with_eeg/total_generated*100:.1f}%")
    print("\nProcessamento concluído!")

if __name__ == "__main__":
    generate_all_graphs_with_eeg()