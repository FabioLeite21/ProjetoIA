#!/usr/bin/env python3
"""
Script para gerar grafos com dados EEG em arquivos .gml SEPARADOS APENAS para SUBJECT_1.
Cada grafo terá 3 arquivos:
- grafo_principal.gml (eye-tracking + EEG nos nós)
- grafo_principal_features.gml (matriz de features)
- grafo_principal_adjacency.gml (matriz de adjacência)
"""

import sys
import os

# Add the project root to sys.path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import pandas as pd
import networkx as nx
import numpy as np
from src.data.utils import infer_timestamps
from src.data.eeg_interpolation import process_subject_data
from src.data.graph_utils import save_separate_gml_files

def generate_subject1_separate_gml():
    """Gera todos os grafos com dados EEG em arquivos .gml separados APENAS para subject_1."""
    
    print("=== GERAÇÃO DE GRAFOS SEPARADOS - SUBJECT_1 APENAS ===")
    print("Cada trial terá 3 arquivos:")
    print("  • session_X_trial_Y.gml           - Grafo principal (eye-tracking + EEG)")
    print("  • session_X_trial_Y_features.gml  - Matriz de features (legível)")
    print("  • session_X_trial_Y_adjacency.gml - Matriz de adjacência (legível)")
    print()
    
    eye_raw_dir = '../database/raw/Eye_raw/seed_v_eye_feature_raw_excel'
    output_base_dir = '../database/graph'
    
    target_subject_id = 1
    
    total_generated = 0
    total_with_eeg = 0
    total_errors = 0
    total_files_created = 0
    
    sessions = ['Session_1', 'Session_2', 'Session_3']
    
    for session_name in sessions:
        session_number = int(session_name.split('_')[1])
        session_dir = os.path.join(eye_raw_dir, session_name)
        
        if not os.path.exists(session_dir):
            print(f"Aviso: Diretório da sessão não encontrado - {session_dir}")
            continue
        
        print(f"Processando {session_name.replace('_', ' ')}...")
        
        target_files = []
        for file_name in sorted(os.listdir(session_dir)):
            if not file_name.endswith('.xlsx') or file_name.startswith('.'):
                continue
            
            parts = file_name.split('_')
            if len(parts) < 3:
                continue
                
            subject_id = int(parts[0])
            
            if subject_id == target_subject_id:
                target_files.append(file_name)
        
        if not target_files:
            print(f"  Nenhum arquivo encontrado para subject {target_subject_id}")
            continue
        
        for file_name in target_files:
            parts = file_name.split('_')
            subject_id = int(parts[0])
            session_id = int(parts[1])
            
            print(f"  Sujeito {subject_id} (Session {session_id}): ", end="", flush=True)
            
            subject_dir = os.path.join(output_base_dir, f'subject_{subject_id}')
            os.makedirs(subject_dir, exist_ok=True)
            
            excel_path = os.path.join(session_dir, file_name)
            
            try:
                xls = pd.ExcelFile(excel_path)
                sheet_names = xls.sheet_names
                
                trials_processed = 0
                trials_with_eeg = 0
                files_created_this_subject = 0
                
                for trial_idx, sheet_name in enumerate(sheet_names):
                    trial_number = trial_idx + 1
                    
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
                            continue
                        
                        df_events = infer_timestamps(df_events)
                        
                        eeg_timestamps, eeg_segments, eeg_labels = process_subject_data(
                            subject_id, session_id, trial_number
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
                        
                        graph_file = os.path.join(subject_dir, f'session_{session_id}_trial_{trial_number}.gml')
                        saved_files = save_separate_gml_files(G, graph_file)
                        
                        files_created_this_trial = len(saved_files)
                        files_created_this_subject += files_created_this_trial
                        total_files_created += files_created_this_trial
                        
                        trials_processed += 1
                        total_generated += 1
                        
                        if eeg_segments:
                            trials_with_eeg += 1
                            total_with_eeg += 1
                        
                    except Exception as e:
                        print(f"Erro no trial {trial_number}: {str(e)[:30]}...")
                        total_errors += 1
                        continue
                
                print(f"{trials_processed} trials processados ({trials_with_eeg} com EEG), {files_created_this_subject} arquivos criados")
                
            except Exception as e:
                print(f"erro no processamento - {str(e)[:50]}...")
                total_errors += 1
    
    print(f"\n=== RESUMO DO PROCESSAMENTO - SUBJECT {target_subject_id} ===")
    print(f"• {total_generated} grafos gerados no total")
    print(f"• {total_with_eeg} grafos incluem dados EEG")
    print(f"• {total_files_created} arquivos .gml criados no total")
    print(f"• {total_errors} trials apresentaram erros")
    if total_generated > 0:
        print(f"• Taxa de sucesso na integração EEG: {total_with_eeg/total_generated*100:.1f}%")
        avg_files_per_trial = total_files_created / total_generated
        print(f"• Média de arquivos por trial: {avg_files_per_trial:.1f}")
    
    print(f"\n=== ARQUIVOS CRIADOS PARA SUBJECT {target_subject_id} ===")
    subject_dir = os.path.join(output_base_dir, f'subject_{target_subject_id}')
    if os.path.exists(subject_dir):
        all_files = sorted(os.listdir(subject_dir))
        
        main_files = [f for f in all_files if not ('_features' in f or '_adjacency' in f)]
        feature_files = [f for f in all_files if '_features.gml' in f]
        adjacency_files = [f for f in all_files if '_adjacency.gml' in f]
        
        print(f"Grafos principais: {len(main_files)}")
        print(f"Arquivos de features: {len(feature_files)}")
        print(f"Arquivos de adjacência: {len(adjacency_files)}")
        
        complete_trials = 0
        incomplete_trials = []
        
        for main_file in main_files:
            if main_file.endswith('.gml'):
                base_name = main_file.replace('.gml', '')
                features_file = f"{base_name}_features.gml"
                adjacency_file = f"{base_name}_adjacency.gml"
                
                if features_file in feature_files and adjacency_file in adjacency_files:
                    complete_trials += 1
                else:
                    incomplete_trials.append(base_name)
        
        print(f"Trials completos (3 arquivos): {complete_trials}")
        if incomplete_trials:
            print(f"Trials incompletos: {len(incomplete_trials)}")
            for trial in incomplete_trials[:5]:
                print(f"  - {trial}")
            if len(incomplete_trials) > 5:
                print(f"  ... e mais {len(incomplete_trials) - 5}")
    
    print("\n=== ESTRUTURA DOS ARQUIVOS GERADOS ===")
    print("Para cada trial completo, foram criados:")
    print("  session_X_trial_Y.gml           - Grafo principal (eye-tracking + EEG)")
    print("  session_X_trial_Y_features.gml  - Matriz de features (legível)")
    print("  session_X_trial_Y_adjacency.gml - Matriz de adjacência (legível)")
    print(f"\nProcessamento do Subject {target_subject_id} concluído!")

if __name__ == "__main__":
    generate_subject1_separate_gml()
