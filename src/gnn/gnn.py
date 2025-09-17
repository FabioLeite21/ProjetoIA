import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from spektral.layers import GlobalAvgPool
from src.gnn.custom_layers import CustomGCNConv, CustomGlobalAvgPool, gcn_filter
from spektral.data import DisjointLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Configurar GPU de forma robusta
def configure_gpu():
    """Configura GPU se disponível, senão usa CPU."""
    print("Configurando dispositivos de processamento...")
    
    try:
        # Listar dispositivos GPU disponíveis
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"GPU(s) detectada(s): {len(gpus)}")
            
            # Configurar growth de memória para evitar pré-alocação
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  - {gpu.name}: Memory growth habilitado")
                except RuntimeError as e:
                    print(f"  - Erro ao configurar {gpu.name}: {e}")
            
            # Verificar se CUDA está disponível
            if tf.test.is_built_with_cuda():
                print("TensorFlow compilado com suporte CUDA")
                if tf.test.is_gpu_available():
                    print("GPU disponível para uso")
                else:
                    print("GPU detectada mas não disponível para uso")
            
            return True
            
        else:
            print("Nenhuma GPU detectada")
            print("Usando CPU para treinamento")
            return False
            
    except Exception as e:
        print(f"Erro na configuração de GPU: {e}")
        print("Continuando com CPU...")
        return False

# Executar configuração
gpu_available = configure_gpu()

# Adicionar o diretório pai ao path para importações
import sys
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from src.data.emotion_graph_dataset import EmotionGraphDataset
from src.data.dataset_splits import create_emotion_splits
from src.gnn.metrics import evaluate_model_metrics, print_metrics_report

# Classe do modelo GNN simplificado para dataset pequeno
class SeedGCNModel(keras.Model):
    def __init__(self, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        # Arquitetura balanceada - 2 camadas GCN otimizadas
        # Normalização corrigida permite arquitetura mais robusta

        # Primeira camada: 317 -> 64
        self.gcn1 = CustomGCNConv(64, activation='relu')
        self.dropout1 = layers.Dropout(0.1)  # Dropout muito baixo

        # Segunda camada: 64 -> 32
        self.gcn2 = CustomGCNConv(32, activation='relu')

        # Camada de pooling global
        self.global_pool = CustomGlobalAvgPool()

        # Saída direta
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def build(self, input_shape):
        """Build method to initialize layer weights."""
        # O build será chamado automaticamente quando necessário
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Pipeline balanceado - 2 camadas GCN

        # Primeira convolução
        h1 = self.gcn1(inputs)
        h1 = self.dropout1(h1, training=training)

        # Segunda convolução
        if len(inputs) == 3:
            x, a, i = inputs
            h2 = self.gcn2([h1, a])
        else:
            x, a = inputs
            h2 = self.gcn2([h1, a])

        # Pooling global
        if len(inputs) == 3:
            pooled = self.global_pool([h2, i])
        else:
            pooled = self.global_pool(h2)

        # Saída direta
        return self.output_layer(pooled)

# Classe wrapper para gerenciar o modelo
class EmotionGNN:
    def __init__(self, input_features=317, num_classes=5):
        self.input_features = input_features  # Número de features por nó (317: 7 eye-tracking + 310 EEG)
        self.num_classes = num_classes  # Número de classes (5 emoções)
        self.model = None  # Modelo será inicializado depois

    def build_model(self):
        """Constrói o modelo GNN."""
        if self.model is not None:
            print("Modelo já existe. Recriando...")
        self.model = SeedGCNModel(num_classes=self.num_classes)
        return self.model

    def compile_model(self, learning_rate=0.001, class_weights=None):
        """Compila o modelo com otimizador e métricas."""
        if self.model is None:
            raise ValueError("Modelo deve ser construído antes de compilar.")

        # Definir métricas para classificação sparse categorical
        precision_metric = keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')

        # Usar weighted categorical crossentropy se class_weights fornecido
        if class_weights is not None:
            loss = self._weighted_categorical_crossentropy(class_weights)
        else:
            loss = 'sparse_categorical_crossentropy'

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,  # Para labels inteiros (0-4)
            metrics=['accuracy', precision_metric]
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

    def _weighted_categorical_crossentropy(self, class_weights):
        """
        Cria uma função de loss com pesos de classe.

        Args:
            class_weights (dict): Dicionário com pesos por classe

        Returns:
            function: Função de loss customizada
        """
        def loss(y_true, y_pred):
            # Converter class_weights dict para tensor
            weights_tensor = tf.constant([class_weights.get(i, 1.0) for i in range(self.num_classes)], dtype=tf.float32)

            # Calcular pesos para cada amostra baseado no label verdadeiro
            sample_weights = tf.gather(weights_tensor, tf.cast(y_true, tf.int32))

            # Calcular cross entropy padrão
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

            # Aplicar pesos
            weighted_loss = ce_loss * sample_weights

            return tf.reduce_mean(weighted_loss)

        return loss

    def calculate_class_weights(self, labels):
        """
        Calcula pesos balanceados para as classes.

        Args:
            labels (list): Lista de labels

        Returns:
            dict: Dicionário com pesos por classe
        """
        import numpy as np

        # Obter classes únicas e calcular pesos
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=labels
        )

        # Converter para dicionário
        class_weight_dict = {int(cls): weight for cls, weight in zip(unique_classes, class_weights)}

        print("Pesos de classe calculados:")
        emotion_names = ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy']
        for cls, weight in class_weight_dict.items():
            if cls < len(emotion_names):
                print(f"  {emotion_names[cls]} (classe {cls}): {weight:.3f}")

        return class_weight_dict

    def train_model(self):
        """Treina o modelo usando o dataset e loaders."""
        try:
            # Carregar o dataset principal
            base_dataset = EmotionGraphDataset('database/graph')
            print(f"Total de grafos carregados: {len(base_dataset.read())}")
        except Exception as e:
            print(f"Erro ao carregar o dataset: {e}")
            return None

        # Criar splits usando a nova abordagem
        try:
            train_dataset, val_dataset, test_dataset = create_emotion_splits(
                base_dataset, test_size=0.2, val_size=0.1, random_state=42
            )

            # Calcular pesos de classe baseados no conjunto de treino
            train_labels = [base_dataset.labels[i] for i in train_dataset.indices]
            class_weights = self.calculate_class_weights(train_labels)

            # Recompilar modelo com pesos de classe e learning rate menor
            print("Recompilando modelo com balanceamento de classes...")
            self.compile_model(learning_rate=0.0001, class_weights=class_weights)  # LR menor para dataset pequeno

        except Exception as e:
            print(f"Erro ao criar splits: {e}")
            return None

        # Criar loaders com batch size menor para dataset pequeno
        try:
            train_loader = DisjointLoader(train_dataset, batch_size=8, epochs=None, shuffle=True)  # Batch size menor
            val_loader = DisjointLoader(val_dataset, batch_size=8, epochs=None) if val_dataset else None
            test_loader = DisjointLoader(test_dataset, batch_size=8, epochs=1)  # Teste usa epochs=1
        except Exception as e:
            print(f"Erro ao criar loaders: {e}")
            return None

        # Configurar callbacks com métricas disponíveis
        callbacks = []
        
        # Early stopping e reduce LR mais agressivos para dataset pequeno
        if val_loader:
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    mode='max',
                    patience=5,  # Patience menor para parar overfitting rapidamente
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    mode='max',
                    factor=0.5,
                    patience=3,  # Reduzir LR mais rapidamente
                    min_lr=1e-8,
                    verbose=1
                )
            ])
        else:
            # Se não tivermos validação, usar apenas loss do treino
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor='loss',
                    mode='min',
                    patience=8,  # Patience menor
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='loss',
                    mode='min',
                    factor=0.5,
                    patience=4,  # Reduzir LR mais rapidamente
                    min_lr=1e-8,
                    verbose=1
                )
            ])

        # Treinar usando o método oficial do Spektral
        try:
            print("Iniciando treinamento com Spektral DisjointLoader...")
            
            # Usar o método .load() para compatibilidade com TensorFlow 2.4+
            validation_data = val_loader.load() if val_loader else None
            
            history = self.model.fit(
                train_loader.load(),
                validation_data=validation_data,
                epochs=30,  # Menos épocas para evitar overfitting
                steps_per_epoch=train_loader.steps_per_epoch,
                validation_steps=val_loader.steps_per_epoch if val_loader else None,
                callbacks=callbacks,
                verbose=1
            )
            
            # Avaliar no conjunto de teste com métricas básicas
            print("\nAvaliando no conjunto de teste...")
            test_results = self.model.evaluate(
                test_loader.load(),
                steps=test_loader.steps_per_epoch,
                verbose=1
            )
            print(f"Acurácia no teste: {test_results[1] * 100:.2f}%")
            
            # Avaliar com métricas detalhadas
            detailed_metrics = evaluate_model_metrics(self.model, test_loader)
            print_metrics_report(detailed_metrics)
            
            # Adicionado por: Davi Augusto - Coletar predições para gráficos
            # Coletar predições do conjunto de teste para gráficos
            test_predictions = []
            test_targets = []
            
            # Reinicializar o loader para coleta de predições
            test_loader_for_pred = DisjointLoader(test_dataset, batch_size=8, epochs=1)
            
            for batch in test_loader_for_pred:
                inputs, targets = batch
                predictions = self.model(inputs, training=False)
                
                # Converter para numpy
                pred_classes = tf.argmax(predictions, axis=1).numpy()
                target_classes = targets.numpy() if hasattr(targets, 'numpy') else targets
                
                test_predictions.extend(pred_classes.tolist())
                test_targets.extend(target_classes.tolist())
            
        except Exception as e:
            print(f"Erro durante o treinamento: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Retornar resultados
        return {
        'history': history,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'test_predictions': test_predictions,  # Adicionado por: Davi Augusto
        'test_targets': test_targets,  # Adicionado por: Davi Augusto
        'detailed_metrics': detailed_metrics,  # Adicionado por: Davi Augusto
        'test_loss': test_loss,  # Adicionado por: Davi Augusto
        'test_accuracy': test_accuracy  # Adicionado por: Davi Augusto
    }

