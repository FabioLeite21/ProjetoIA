import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from spektral.layers import GCNConv, GlobalAvgPool
from spektral.data import DisjointLoader
from sklearn.model_selection import train_test_split
from data.emotion_graph_dataset import EmotionGraphDataset

# Classe do modelo GNN
class SeedGCNModel(keras.Model):
    def __init__(self, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        # Primeira camada de convolução: 263 -> 32
        self.gcn1 = GCNConv(32, activation='relu')
        # Segunda camada de convolução: 32 -> 64
        self.gcn2 = GCNConv(64, activation='relu')
        # Camada de pooling global
        self.global_pool = GlobalAvgPool()
        # Camada densa intermediária: 64 -> 64
        self.dense = layers.Dense(64, activation='relu')
        # Dropout de 50% para regularização
        self.dropout = layers.Dropout(0.5)
        # Camada de saída: 64 -> 5 (softmax para 5 classes)
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x, a = inputs  # x = features, a = matriz de adjacência
        h1 = self.gcn1([x, a])  # Primeira convolução
        h2 = self.gcn2([h1, a])  # Segunda convolução
        pooled = self.global_pool(h2)  # Pooling global
        dense_output = self.dense(pooled)  # Camada densa
        dropout_out = self.dropout(dense_output, training=training)  # Dropout no treino
        return self.output_layer(dropout_out)  # Saída com probabilidades

# Classe wrapper para gerenciar o modelo
class EmotionGNN:
    def __init__(self, input_features=263, num_classes=5):
        self.input_features = input_features  # Número de features por nó (263)
        self.num_classes = num_classes  # Número de classes (5 emoções)
        self.model = None  # Modelo será inicializado depois

    def build_model(self):
        """Constrói o modelo GNN."""
        if self.model is not None:
            print("Modelo já existe. Recriando...")
        self.model = SeedGCNModel(num_classes=self.num_classes)
        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compila o modelo com otimizador e métricas."""
        if self.model is None:
            raise ValueError("Modelo deve ser construído antes de compilar.")
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',  # Para labels inteiros (0-4)
            metrics=['accuracy', 'precision', 'recall', 'f1_score']  # Métricas adicionais
        )

    def model_summary(self):
        """Exibe o resumo do modelo."""
        if self.model is None:
            raise ValueError("Modelo deve ser construído antes de exibir o resumo.")
        self.model.summary()

    def save_model(self, filepath):
        """Salva o modelo treinado."""
        if self.model is None:
            raise ValueError("Modelo deve existir para ser salvo.")
        self.model.save(filepath)
        print(f"Modelo salvo em {filepath}")

    def load_model(self, filepath):
        """Carrega um modelo previamente salvo."""
        self.model = keras.models.load_model(filepath)
        print(f"Modelo carregado de {filepath}")

    def train_model(self):
        """Treina o modelo usando o dataset e loaders."""
        try:
            # Carregar o dataset
            dataset = EmotionGraphDataset('database/graph')
            print(f"Total de grafos carregados: {len(dataset)}")
        except Exception as e:
            print(f"Erro ao carregar o dataset: {e}")
            return None

        # Dividir em treino, validação e teste (80/10/10)
        train_dataset, test_dataset = train_test_split(
            dataset, test_size=0.2, random_state=42
        )
        val_dataset, test_dataset = train_test_split(
            test_dataset, test_size=0.5, random_state=42
        )
        print(f"Treino: {len(train_dataset)}, Validação: {len(val_dataset)}, Teste: {len(test_dataset)}")

        # Criar loaders
        try:
            train_loader = DisjointLoader(train_dataset, batch_size=32, epochs=1, shuffle=True)
            val_loader = DisjointLoader(val_dataset, batch_size=32, epochs=1)
            test_loader = DisjointLoader(test_dataset, batch_size=32, epochs=1)
        except Exception as e:
            print(f"Erro ao criar loaders: {e}")
            return None

        # Definir callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]

        # Treinar o modelo
        try:
            history = self.model.fit(
                train_loader.load(),
                validation_data=val_loader.load(),
                epochs=50,
                steps_per_epoch=train_loader.steps_per_epoch,
                validation_steps=val_loader.steps_per_epoch,
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e:
            print(f"Erro durante o treinamento: {e}")
            return None

        # Avaliar no conjunto de teste
        try:
            test_results = self.model.evaluate(
                test_loader.load(),
                steps=test_loader.steps_per_epoch,
                verbose=1
            )
            print(f"Acurácia no teste: {test_results[1] * 100:.2f}%")
        except Exception as e:
            print(f"Erro durante a avaliação: {e}")

        # Retornar resultados
        return {
            'history': history,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        }

