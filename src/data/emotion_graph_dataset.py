from spektral.data import Dataset, Graph
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

class EmotionGraphDataset(Dataset):
    def __init__(self, graph_dir, labels_excel_path=None, **kwargs):
        self.graph_dir = graph_dir
        if labels_excel_path is None:
            self.labels_excel_path = os.path.join(project_root, 'database', 'emotion_label_and_stimuli_order.xlsx')
        else:
            if not os.path.isabs(labels_excel_path):
                self.labels_excel_path = os.path.join(project_root, labels_excel_path)
            else:
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
        # Verificar se o arquivo existe
        if not os.path.exists(self.labels_excel_path):
            raise FileNotFoundError(f"Arquivo de labels não encontrado: {self.labels_excel_path}")
        
        # Carregar o Excel (assumindo uma única sheet com os dados)
        df = pd.read_excel(self.labels_excel_path, header=None)
        
        # mapeamento de emoções para labels numéricos
        emotion_to_label = {'Disgust': 0, 'Fear': 1, 'Sad': 2, 'Neutral': 3, 'Happy': 4}
        
        session_maps = {}
        current_session = None
        
        for i, row in df.iterrows():
            row_data = row.dropna().tolist()
            if not row_data:
                continue
                
            # Linha do cabeçalho - Session 1 está no índice 1
            if len(row_data) > 0 and 'Movie orders for three sessions' in str(row_data[0]):
                if len(row_data) > 1 and 'Session 1' in str(row_data[1]):
                    # Pegar as emoções da Session 1 (índices 2 em diante)
                    emotions = row_data[2:]
                    labels = [emotion_to_label.get(e, -1) for e in emotions]
                    session_maps['Session 1'] = labels
                    print(f"Mapeamento carregado para Session 1: {labels}")
                continue
            
            # Linhas das outras sessões
            if len(row_data) > 0 and str(row_data[0]).startswith('Session'):
                session_name = str(row_data[0]).strip()
                emotions = row_data[1:]  # Emoções começam no índice 1
                labels = [emotion_to_label.get(e, -1) for e in emotions]
                session_maps[session_name] = labels
                print(f"Mapeamento carregado para {session_name}: {labels}")
        
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
            df_features = pd.read_csv(feature_matrix_path)
            X = df_features.values  # Converte para numpy array
            # Normalização
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        else:
            print(f"Arquivo de features não encontrado: {feature_matrix_path}. Usando array vazio.")
            X = np.zeros((7, 310))  # Placeholder com 317 features (7 eye-tracking + 310 EEG)

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