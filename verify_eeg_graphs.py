"""
Script para verificar se os grafos foram gerados com dados EEG.
"""

import os
import networkx as nx
from src.data.graph_utils import load_graph_with_eeg

def verify_eeg_graphs():
    """Verifica se os grafos contêm dados EEG."""
    
    print("=== VERIFICANDO GRAFOS COM DADOS EEG ===\n")
    
    # Verificar alguns grafos de exemplo
    test_graphs = [
        'database/graph/subject_1/session_1_trial_1.gml',
        'database/graph/subject_1/session_1_trial_2.gml',
        'database/graph/subject_2/session_1_trial_1.gml',
    ]
    
    for graph_path in test_graphs:
        print(f"Verificando: {graph_path}")
        
        if not os.path.exists(graph_path):
            print(f"  [!] Arquivo não encontrado")
            continue
        
        try:
            # Carregar grafo
            G = nx.read_gml(graph_path)
            
            # Verificar primeiro nó
            first_node = list(G.nodes(data=True))[0]
            eeg_data = first_node[1].get('eeg_data', 'NA')
            
            if eeg_data != 'NA' and len(eeg_data) > 10:
                print(f"  [OK] Contém dados EEG ({len(eeg_data)} chars)")
                
                # Testar carregamento completo
                _, eeg_matrix, _ = load_graph_with_eeg(graph_path)
                if eeg_matrix is not None:
                    print(f"       Matriz EEG: {eeg_matrix.shape}")
                else:
                    print(f"       [!] Erro ao carregar matriz EEG")
            else:
                print(f"  [!] SEM dados EEG")
                
        except Exception as e:
            print(f"  [ERRO] {e}")
        
        print()

def count_generated_graphs():
    """Conta quantos grafos foram gerados."""
    
    print("=== CONTAGEM DE GRAFOS GERADOS ===\n")
    
    graph_dir = 'database/graph'
    
    if not os.path.exists(graph_dir):
        print(f"Diretório {graph_dir} não existe!")
        return
    
    total_graphs = 0
    subjects_with_graphs = 0
    
    for subject_folder in os.listdir(graph_dir):
        subject_path = os.path.join(graph_dir, subject_folder)
        
        if os.path.isdir(subject_path) and subject_folder.startswith('subject_'):
            subject_graphs = [f for f in os.listdir(subject_path) if f.endswith('.gml')]
            
            if subject_graphs:
                subjects_with_graphs += 1
                total_graphs += len(subject_graphs)
                
                print(f"{subject_folder}: {len(subject_graphs)} grafos")
    
    print(f"\nResumo:")
    print(f"- {subjects_with_graphs} sujeitos processados")
    print(f"- {total_graphs} grafos gerados")
    print(f"- Esperado: ~45 grafos por sujeito (3 sessões × 15 trials)")

if __name__ == "__main__":
    count_generated_graphs()
    print()
    verify_eeg_graphs()