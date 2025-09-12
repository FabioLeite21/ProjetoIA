from spektral.data import Dataset, Graph
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class EmotionGraphDataset(Dataset):
    def __init__(self, graph_dir, labels_excel_path='emotion_label_and_stimuli_order.xlsx', **kwargs):
        self.graph_dir = graph_dir
        self.labels_excel_path = labels_excel_path
        self.graph_list = []
        self.labels = []
        self.session_emotion_maps = self.load_emotion_maps()  # carrega mapeamentos de emoções por sessão
        super().__init__(**kwargs)
        self.load_graphs()

    def load_emotion_maps(self):
        """
        Carrega os mapeamentos de emoções por sessão a partir do Excel.
        Retorna um dicionário com chaves 'Session 1', 'Session 2', 'Session 3' e listas de labels (0-4).
        """
        # carregar o excel 
        df = pd.read_excel(self.labels_excel_path, header=None)
        
        # mapeamento de emoções para labels numéricos
        emotion_to_label = {'Disgust': 0, 'Fear': 1, 'Sad': 2, 'Neutral': 3, 'Happy': 4}
        
        session_maps = {}
        current_session = None
        
        for i, row in df.iterrows():
            if pd.notna(row[0]) and 'Movie orders for three sessions' in str(row[0]):
                continue  
            if pd.notna(row[0]) and 'Session' in str(row[0]):
                current_session = str(row[0]).strip()
                emotions = row[1:].dropna().tolist()  
                labels = [emotion_to_label.get(e, -1) for e in emotions]  
                session_maps[current_session] = labels
                print(f"Mapeamento carregado para {current_session}: {labels}")
        
        return session_maps

    def load_graphs(self):
        """
        Carrega todos os arquivos .gml e processa em objetos Graph, usando matrizes CSV existentes.
        """
        gml_files = []
        for root, dirs, files in os.walk(self.graph_dir):
            for file in files:
                if file.endswith('.gml'):
                    gml_files.append(os.path.join(root, file))
        
        for gml_file in gml_files:
            X, A = self.process_gml(gml_file)
            label = self.extract_label(gml_file)
            graph = Graph(x=X, a=A, y=label)
            self.graph_list.append(graph)
            self.labels.append(label)

    def process_gml(self, gml_file):
        """
        Processa o arquivo .gml e lê as matrizes de adjacência e features de arquivos CSV correspondentes.
        """
        base_name = os.path.splitext(gml_file)[0]  # Ex.: 'database/graph/session_1_trial_1'
        adj_matrix_path = f"{base_name}_adjacency_matrix.csv"
        feature_matrix_path = f"{base_name}_feature_matrix.csv"

        # carrega a matriz de adjacência
        if os.path.exists(adj_matrix_path):
            A = np.loadtxt(adj_matrix_path, delimiter=',')
        else:
            print(f"Arquivo de adjacência não encontrado: {adj_matrix_path}. Usando matriz vazia.")
            A = np.zeros((1, 1)) 

        # carrega a matriz de features
        if os.path.exists(feature_matrix_path):
            X = np.loadtxt(feature_matrix_path, delimiter=',')
            # normalização
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        else:
            print(f"Arquivo de features não encontrado: {feature_matrix_path}. Usando array vazio.")
            X = np.zeros((1, 263))  # placeholder com 263 features (7 eye + 256 EEG)

        return X, A

    def extract_label(self, gml_file):
        """
        Extrai o label da emoção baseado no nome do arquivo e mapeamentos de sessões.
        """
        filename = os.path.basename(gml_file)
        parts = filename.split('_')
        if len(parts) < 4:
            print(f"Nome de arquivo inválido: {filename}. Usando label default 0.")
            return 0
        
        session_str = parts[1]  # 'session_X'
        trial_str = parts[3].split('.')[0]  # 'trial_Y.gml' -> 'Y'
        
        session_id = int(session_str.replace('session', ''))
        trial_idx = int(trial_str) - 1  # 1-indexed para 0-indexed
        
        session_key = f'Session {session_id}'
        if session_key in self.session_emotion_maps:
            emotions = self.session_emotion_maps[session_key]
            if 0 <= trial_idx < len(emotions):
                return emotions[trial_idx]
            else:
                print(f"Trial index {trial_idx} fora do range para {session_key}. Usando label default 0.")
        
        print(f"Sessão {session_key} não encontrada. Usando label default 0.")
        return 0

    def read(self):
        return self.graph_list

# exemplo:  
if __name__ == "__main__":
    dataset = EmotionGraphDataset('database/graph')
    print(f"Total de grafos carregados: {len(dataset)}")
    
    # Split train/val/test (80/10/10)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)
    
    print(f"Treino: {len(train_dataset)}, Validação: {len(val_dataset)}, Teste: {len(test_dataset)}")
    
    from spektral.data import DisjointLoader
    train_loader = DisjointLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DisjointLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DisjointLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Loaders prontos para treinamento!")