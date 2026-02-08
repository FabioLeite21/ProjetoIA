"""
Modelo MLP baseline para classificação de emoções que ignora a estrutura do grafo.
Útil para comparação com GNNs e verificar se a estrutura do grafo está ajudando.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler


class EmotionMLPBaseline(keras.Model):
    """
    Modelo MLP simples que trata cada grafo como um conjunto de features agregadas.
    """

    def __init__(self, input_features=317, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.input_features = input_features

        # Arquitetura MLP simples
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.3)

        self.dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.3)

        self.dense3 = layers.Dense(32, activation='relu')
        self.dropout3 = layers.Dropout(0.2)

        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        x = self.dense3(x)
        x = self.dropout3(x, training=training)

        return self.output_layer(x)


class EmotionMLPTrainer:
    """
    Classe para treinar o modelo MLP baseline.
    """

    def __init__(self, input_features=317, num_classes=5):
        self.input_features = input_features
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self):
        """Constrói o modelo MLP."""
        self.model = EmotionMLPBaseline(
            input_features=self.input_features,
            num_classes=self.num_classes
        )
        return self.model

    def compile_model(self, learning_rate=0.001, class_weights=None):
        """Compila o modelo."""
        if class_weights is not None:
            loss = self._weighted_categorical_crossentropy(class_weights)
        else:
            loss = 'sparse_categorical_crossentropy'

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy']
        )

    def _weighted_categorical_crossentropy(self, class_weights):
        """Cria função de loss com pesos de classe."""
        def loss(y_true, y_pred):
            weights_tensor = tf.constant([class_weights.get(i, 1.0) for i in range(self.num_classes)], dtype=tf.float32)
            sample_weights = tf.gather(weights_tensor, tf.cast(y_true, tf.int32))
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            weighted_loss = ce_loss * sample_weights
            return tf.reduce_mean(weighted_loss)
        return loss

    def prepare_data_from_graphs(self, dataset):
        """
        Converte grafos em features agregadas para MLP.

        Args:
            dataset: Dataset de grafos

        Returns:
            tuple: (X, y) onde X são features agregadas e y são labels
        """
        X = []
        y = []

        graphs = dataset.read()
        print(f"Carregados {len(graphs)} grafos do dataset")

        for i, graph in enumerate(graphs):
            try:
                # Agregar features dos nós usando estatísticas descritivas
                node_features = graph.x  # Shape: (n_nodes, n_features)

                if node_features.shape[0] == 0:
                    # Grafo vazio - usar zeros
                    aggregated_features = np.zeros(self.input_features * 8)  # Expandido para mais estatísticas
                else:
                    # Calcular estatísticas agregadas expandidas
                    mean_features = np.mean(node_features, axis=0)
                    std_features = np.std(node_features, axis=0)
                    min_features = np.min(node_features, axis=0)
                    max_features = np.max(node_features, axis=0)

                    # Estatísticas adicionais
                    median_features = np.median(node_features, axis=0)
                    q25_features = np.percentile(node_features, 25, axis=0)
                    q75_features = np.percentile(node_features, 75, axis=0)
                    range_features = max_features - min_features

                    # Concatenar todas as estatísticas
                    aggregated_features = np.concatenate([
                        mean_features, std_features, min_features, max_features,
                        median_features, q25_features, q75_features, range_features
                    ])

                X.append(aggregated_features)
                y.append(graph.y)

            except Exception as e:
                print(f"ERRO no grafo {i}: {e}")
                continue

        return np.array(X), np.array(y)

    def calculate_class_weights(self, y):
        """Calcula pesos balanceados para as classes."""
        unique_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight_dict = {int(cls): weight for cls, weight in zip(unique_classes, class_weights)}

        print("Pesos de classe calculados (MLP):")
        emotion_names = ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy']
        for cls, weight in class_weight_dict.items():
            if cls < len(emotion_names):
                print(f"  {emotion_names[cls]} (classe {cls}): {weight:.3f}")

        return class_weight_dict

    def train_model(self, train_dataset, val_dataset=None, test_dataset=None):
        """
        Treina o modelo MLP baseline.

        Args:
            train_dataset: Dataset de grafos de treino (pode incluir augmentation)
            val_dataset: Dataset de grafos de validação (sem augmentation)
            test_dataset: Dataset de grafos de teste (sem augmentation)

        Returns:
            dict: Resultados do treinamento
        """
        print("Preparando dados para MLP baseline...")

        # Converter grafos de treino em features agregadas
        X_train, y_train = self.prepare_data_from_graphs(train_dataset)

        if len(X_train) == 0:
            print("ERRO: Nenhum grafo foi processado pelo dataset de treino")
            return None

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        print(f"Treino convertido: {X_train.shape[0]} amostras, {X_train.shape[1]} features")

        if val_dataset is not None and test_dataset is not None:
            # Datasets já divididos externamente (sem vazamento)
            X_val, y_val = self.prepare_data_from_graphs(val_dataset)
            X_test, y_test = self.prepare_data_from_graphs(test_dataset)
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            # Normalizar: fit APENAS no treino, transform em todos
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
        else:
            # Fallback legado: split interno (sem augmentation externa)
            print("AVISO: val/test datasets nao fornecidos, usando split interno")
            X_train = self.scaler.fit_transform(X_train)
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
            )

        print(f"Splits - Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")

        # Calcular pesos de classe
        class_weights = self.calculate_class_weights(y_train)

        # Compilar modelo
        self.compile_model(learning_rate=0.001, class_weights=class_weights)

        # Configurar callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                mode='max',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        print("Iniciando treinamento do MLP baseline...")

        # Treinar modelo
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )

        # Avaliar no conjunto de teste
        print("\nAvaliando MLP baseline no conjunto de teste...")
        test_results = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"Acurácia no teste (MLP): {test_results[1] * 100:.2f}%")

        # Predições detalhadas
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        return {
            'history': history,
            'test_accuracy': test_results[1],
            'y_true': y_test,
            'y_pred': y_pred_classes,
            'X_test': X_test
        }

    def save_model(self, filepath):
        """Salva o modelo treinado."""
        if self.model is None:
            raise ValueError("Modelo deve existir para ser salvo.")
        self.model.save(filepath)
        print(f"Modelo MLP salvo em {filepath}")


def compare_with_gnn():
    """
    Função para comparar resultados do MLP com GNN.
    """
    print("=" * 60)
    print("COMPARAÇÃO MLP BASELINE vs GNN")
    print("=" * 60)
    print("O MLP baseline ignora completamente a estrutura do grafo.")
    print("Se o MLP tiver acurácia similar ou superior à GNN, isso indica que:")
    print("1. A estrutura do grafo não está ajudando")
    print("2. As features agregadas já contêm informação suficiente")
    print("3. A conectividade dos grafos pode estar inadequada")
    print("=" * 60)


if __name__ == "__main__":
    # Exemplo de uso
    import sys
    import os

    # Adicionar o diretório do projeto ao path
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    sys.path.insert(0, project_root)

    from src.data.emotion_graph_dataset import EmotionGraphDataset

    # Carregar dataset
    dataset = EmotionGraphDataset('database/graph')

    # Treinar MLP baseline
    mlp_trainer = EmotionMLPTrainer()
    mlp_trainer.build_model()
    results = mlp_trainer.train_model(dataset)

    compare_with_gnn()