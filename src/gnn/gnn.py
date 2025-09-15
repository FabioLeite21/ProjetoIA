import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from spektral.layers import GlobalAvgPool
from src.gnn.custom_layers import CustomGCNConv, CustomGlobalAvgPool, gcn_filter
from spektral.data import DisjointLoader
from sklearn.model_selection import train_test_split

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

# Classe do modelo GNN
class SeedGCNModel(keras.Model):
    def __init__(self, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        # Primeira camada de convolução: 317 -> 32 (usando camada customizada)
        self.gcn1 = CustomGCNConv(32, activation='relu')
        # Segunda camada de convolução: 32 -> 64 (usando camada customizada)
        self.gcn2 = CustomGCNConv(64, activation='relu')
        # Camada de pooling global (usando versão customizada)
        self.global_pool = CustomGlobalAvgPool()
        # Camada densa intermediária: 64 -> 64
        self.dense = layers.Dense(64, activation='relu')
        # Dropout de 50% para regularização
        self.dropout = layers.Dropout(0.5)
        # Camada de saída: 64 -> 5 (softmax para 5 classes)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def build(self, input_shape):
        """Build method to initialize layer weights."""
        # O build será chamado automaticamente quando necessário
        super().build(input_shape)

    def call(self, inputs, training=False):
        # As camadas customizadas lidam com os inputs automaticamente
        # DisjointLoader passa (x, a, i) onde i é o batch index
        
        # Primeira convolução com normalização GCN
        h1 = self.gcn1(inputs)
        
        # Segunda convolução - passar features e adjacência
        if len(inputs) == 3:
            x, a, i = inputs
            h2 = self.gcn2([h1, a])
            # Pooling global com batch index para DisjointLoader
            pooled = self.global_pool([h2, i])
        else:
            x, a = inputs
            h2 = self.gcn2([h1, a])
            # Pooling global simples
            pooled = self.global_pool(h2)
            
        # Camadas densas finais
        dense_output = self.dense(pooled)
        dropout_out = self.dropout(dense_output, training=training)
        return self.output_layer(dropout_out)

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
        # Definir métricas para classificação sparse categorical
        precision_metric = keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',  # Para labels inteiros (0-4)
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
        except Exception as e:
            print(f"Erro ao criar splits: {e}")
            return None

        # Criar loaders usando datasets corretamente estruturados
        # Nota: epochs=None permite que o loader rode indefinidamente
        try:
            train_loader = DisjointLoader(train_dataset, batch_size=32, epochs=None, shuffle=True)
            val_loader = DisjointLoader(val_dataset, batch_size=32, epochs=None) if val_dataset else None
            test_loader = DisjointLoader(test_dataset, batch_size=32, epochs=1)  # Teste usa epochs=1
        except Exception as e:
            print(f"Erro ao criar loaders: {e}")
            return None

        # Configurar callbacks com métricas disponíveis
        callbacks = []
        
        # Early stopping e reduce LR apenas se tivermos validação
        if val_loader:
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',  # Usar accuracy ao invés de loss
                    mode='max',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',  # Usar accuracy ao invés de loss
                    mode='max',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ])
        else:
            # Se não tivermos validação, usar apenas loss do treino
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor='loss',
                    mode='min',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='loss',
                    mode='min',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-7,
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
                epochs=50,
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
            'test_loader': test_loader
        }

