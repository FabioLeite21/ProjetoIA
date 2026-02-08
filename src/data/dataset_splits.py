"""
Classes para criar splits de train/validation/test dos datasets de grafos de emoção.
"""

from spektral.data import Dataset
import numpy as np
from collections.abc import Sequence
from .augmentation import DataAugmenter


class EmotionDatasetSplit(Dataset, Sequence):
    """
    Dataset split que herda corretamente de spektral.data.Dataset e Sequence
    para compatibilidade com DisjointLoader e shuffling.
    """
    
    def __init__(self, base_dataset, indices, **kwargs):
        """
        Args:
            base_dataset: Dataset principal contendo todos os grafos
            indices: Lista de índices para este split
        """
        self.base_dataset = base_dataset
        self.indices = list(indices)  # Garantir que é uma lista
        self._graphs_cache = None  # Cache para evitar recarregar
        super().__init__(**kwargs)
    
    def read(self):
        """
        Retorna os grafos correspondentes aos índices deste split.
        """
        if self._graphs_cache is None:
            all_graphs = self.base_dataset.read()
            self._graphs_cache = [all_graphs[i] for i in self.indices]
        return self._graphs_cache
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Retorna o grafo no índice especificado dentro deste split.
        Implementa corretamente a interface Sequence.
        """
        graphs = self.read()  # Usar cache
        
        if isinstance(idx, slice):
            # Lidar com slices
            return graphs[idx]
        elif isinstance(idx, (list, tuple)):
            # Lidar com listas de índices
            return [graphs[i] for i in idx]
        else:
            # Índice único
            if idx < 0:
                idx = len(graphs) + idx  # Suporte para índices negativos
            if not 0 <= idx < len(graphs):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(graphs)}")
            return graphs[idx]
    
    def __iter__(self):
        """
        Permite iteração sobre o dataset.
        """
        return iter(self.read())
    
    def __contains__(self, item):
        """
        Verifica se um item está no dataset.
        """
        return item in self.read()


class AugmentedEmotionDatasetSplit(Dataset, Sequence):
    """
    Split de treino com data augmentation aplicada apenas aos grafos deste split.
    Compatível com DisjointLoader e iteração direta.
    """

    def __init__(self, base_split, augmentation_config=None, **kwargs):
        """
        Args:
            base_split: EmotionDatasetSplit contendo os grafos originais de treino
            augmentation_config: Dicionário com parâmetros de augmentation
        """
        self.base_split = base_split
        self.augmentation_config = augmentation_config or {}
        self._graphs_cache = None
        super().__init__(**kwargs)

    def read(self):
        """
        Retorna grafos originais + aumentados (apenas do split de treino).
        """
        if self._graphs_cache is None:
            original_graphs = self.base_split.read()

            # Separar parâmetros do augmenter dos parâmetros do dataset
            augmenter_params = {
                'noise_prob': self.augmentation_config.get('noise_prob', 0.7),
                'temporal_prob': self.augmentation_config.get('temporal_prob', 0.5),
                'graph_prob': self.augmentation_config.get('graph_prob', 0.3),
                'noise_std': self.augmentation_config.get('noise_std', 0.05),
                'time_shift_range': self.augmentation_config.get('time_shift_range', 0.1),
                'eye_jitter_std': self.augmentation_config.get('eye_jitter_std', 2.0),
            }

            augmenter = DataAugmenter(**augmenter_params)
            self._graphs_cache = augmenter.augment_dataset(
                original_graphs,
                augmentation_factor=self.augmentation_config.get('augmentation_factor', 2),
                preserve_class_balance=self.augmentation_config.get('preserve_class_balance', True),
            )
        return self._graphs_cache

    def __len__(self):
        return len(self.read())

    def __getitem__(self, idx):
        graphs = self.read()
        if isinstance(idx, slice):
            return graphs[idx]
        elif isinstance(idx, (list, tuple)):
            return [graphs[i] for i in idx]
        else:
            if idx < 0:
                idx = len(graphs) + idx
            if not 0 <= idx < len(graphs):
                raise IndexError(f"Index {idx} fora do range para dataset de tamanho {len(graphs)}")
            return graphs[idx]

    def __iter__(self):
        return iter(self.read())

    def __contains__(self, item):
        return item in self.read()


def create_emotion_splits(base_dataset, test_size=0.2, val_size=0.1, random_state=42):
    """
    Cria splits de train/validation/test a partir de um dataset base.
    
    Args:
        base_dataset: Dataset principal (EmotionGraphDataset)
        test_size: Proporção do conjunto de teste (padrão: 0.2 = 20%)
        val_size: Proporção do conjunto de validação (padrão: 0.1 = 10%)
        random_state: Seed para reprodutibilidade
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    from sklearn.model_selection import train_test_split
    
    # Obter total de amostras
    total_samples = len(base_dataset.read())
    all_indices = list(range(total_samples))
    
    # Obter labels para estratificação
    labels = base_dataset.labels
    
    # Primeiro split: separar teste
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        all_indices, labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    
    # Segundo split: separar validação do treino
    if val_size > 0:
        val_ratio = val_size / (1 - test_size)  # Ajustar proporção para o conjunto reduzido
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            train_val_indices, train_val_labels,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_labels
        )
    else:
        train_indices = train_val_indices
        val_indices = []
    
    # Criar datasets split
    train_dataset = EmotionDatasetSplit(base_dataset, train_indices)
    val_dataset = EmotionDatasetSplit(base_dataset, val_indices) if val_indices else None
    test_dataset = EmotionDatasetSplit(base_dataset, test_indices)
    
    print(f"Splits criados - Treino: {len(train_indices)}, Validação: {len(val_indices)}, Teste: {len(test_indices)}")
    
    return train_dataset, val_dataset, test_dataset