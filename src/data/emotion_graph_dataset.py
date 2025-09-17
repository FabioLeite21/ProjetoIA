from spektral.data import Dataset, Graph
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .augmentation import DataAugmenter

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

class EmotionGraphDataset(Dataset):
    def __init__(self, graph_dir, labels_excel_path=None, normalization='minmax',
                 use_augmentation=False, augmentation_config=None, **kwargs):
        self.graph_dir = graph_dir
        self.normalization = normalization  # 'minmax', 'standard', or 'none'
        self.use_augmentation = use_augmentation
        self.augmentation_config = augmentation_config or {}

        if labels_excel_path is None:
            self.labels_excel_path = os.path.join(project_root, 'database', 'emotion_label_and_stimuli_order.xlsx')
        else:
            if not os.path.isabs(labels_excel_path):
                self.labels_excel_path = os.path.join(project_root, labels_excel_path)
            else:
                self.labels_excel_path = labels_excel_path

        self.graph_list = []
        self.labels = []
        self.original_graph_list = []  # Keep original graphs separate
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

        print(f"Encontrados {len(gml_files)} arquivos GML para processar")

        for gml_file in gml_files:
            try:
                X, A = self.process_gml(gml_file)
                label = self.extract_label(gml_file)

                # Validar dados antes de criar o grafo
                if X.size == 0 or A.size == 0:
                    print(f"Aviso: Dados vazios em {gml_file}, pulando...")
                    continue

                # Verificar consistência entre matriz de features e adjacência
                if X.shape[0] != A.shape[0] or A.shape[0] != A.shape[1]:
                    print(f"Aviso: Inconsistência nas dimensões em {gml_file}")
                    print(f"  Features: {X.shape}, Adjacência: {A.shape}")
                    # Tentar corrigir redimensionando a menor
                    min_nodes = min(X.shape[0], A.shape[0])
                    X = X[:min_nodes]
                    A = A[:min_nodes, :min_nodes]

                graph = Graph(x=X, a=A, y=label)
                self.graph_list.append(graph)
                self.original_graph_list.append(graph)  # Keep original
                self.labels.append(label)

            except Exception as e:
                print(f"Erro ao processar {gml_file}: {e}")
                continue

        print(f"Carregados {len(self.graph_list)} grafos válidos")

        # Apply data augmentation if requested
        if self.use_augmentation and len(self.graph_list) > 0:
            self._apply_augmentation()

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

            # Aplicar normalização baseada no parâmetro
            if self.normalization != 'none' and not self._is_already_normalized(X):
                if self.normalization == 'minmax':
                    scaler = MinMaxScaler()
                elif self.normalization == 'standard':
                    scaler = StandardScaler()
                else:
                    raise ValueError(f"Normalização '{self.normalization}' não suportada")

                X = scaler.fit_transform(X)
                # print(f"Normalização '{self.normalization}' aplicada em {feature_matrix_path}")
            elif self.normalization == 'none':
                pass  # Silencioso quando desabilitado
                # print(f"Normalização desabilitada para {feature_matrix_path}")
            else:
                pass  # Silencioso quando já normalizado
                # print(f"Dados já normalizados em {feature_matrix_path}")

            # Verificar se há valores inválidos
            if np.isnan(X).any() or np.isinf(X).any():
                print(f"Aviso: valores inválidos encontrados em {feature_matrix_path}")
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            print(f"Arquivo de features não encontrado: {feature_matrix_path}. Usando array vazio.")
            X = np.zeros((7, 317))  # Placeholder com 317 features (7 eye-tracking + 310 EEG)

        return X, A

    def _is_already_normalized(self, X, tolerance=0.1):
        """
        Verifica se os dados já estão normalizados (valores aproximadamente entre 0 e 1).

        Args:
            X (np.ndarray): Matriz de features
            tolerance (float): Tolerância para considerar normalizado

        Returns:
            bool: True se já normalizado
        """
        if X.size == 0:
            return True

        min_val = np.min(X)
        max_val = np.max(X)

        # Considerar normalizado se valores estão aproximadamente entre 0 e 1
        return (min_val >= -tolerance and max_val <= 1 + tolerance and
                max_val - min_val > 0.1)  # Deve ter variação mínima

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

    def _apply_augmentation(self):
        """Apply data augmentation to increase dataset size."""
        print(f"\nAplicando data augmentation...")

        # Separate augmenter parameters from dataset parameters
        augmenter_params = {
            'noise_prob': self.augmentation_config.get('noise_prob', 0.7),
            'temporal_prob': self.augmentation_config.get('temporal_prob', 0.5),
            'graph_prob': self.augmentation_config.get('graph_prob', 0.3),
            'noise_std': self.augmentation_config.get('noise_std', 0.05),
            'time_shift_range': self.augmentation_config.get('time_shift_range', 0.1),
            'eye_jitter_std': self.augmentation_config.get('eye_jitter_std', 2.0)
        }

        # Create augmenter with only valid parameters
        augmenter = DataAugmenter(**augmenter_params)

        # Augment the original graphs
        augmented_graphs = augmenter.augment_dataset(
            self.original_graph_list,
            augmentation_factor=self.augmentation_config.get('augmentation_factor', 3),
            preserve_class_balance=self.augmentation_config.get('preserve_class_balance', True)
        )

        # Update graph list and labels
        self.graph_list = augmented_graphs
        self.labels = [graph.y for graph in augmented_graphs]

        print(f"Dataset augmentation aplicada: {len(self.original_graph_list)} -> {len(self.graph_list)} grafos")

    def read(self):
        return self.graph_list