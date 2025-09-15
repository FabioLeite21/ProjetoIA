"""
Camadas GCN customizadas para evitar problemas com máscaras None no Spektral.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spektral.layers import GlobalAvgPool
import tensorflow.keras.backend as K


class CustomGCNConv(layers.Layer):
    """
    Camada de convolução de grafo customizada que evita problemas com máscaras None.
    Baseada na implementação do Spektral GCNConv mas sem problemas de masking.
    """
    
    def __init__(self, channels, activation=None, use_bias=True, **kwargs):
        """
        Args:
            channels: Número de canais de saída
            activation: Função de ativação
            use_bias: Se deve usar bias
        """
        super().__init__(**kwargs)
        self.channels = channels
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.supports_masking = False  # Desabilitar masking explicitamente
        
    def build(self, input_shape):
        """Constrói os pesos da camada."""
        # Lidar com diferentes formatos de input_shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0:
            # input_shape pode ser uma lista/tupla de shapes ou uma tupla de tensores
            if hasattr(input_shape[0], 'shape'):
                # input_shape[0] é um tensor
                input_dim = input_shape[0].shape[-1]
            elif isinstance(input_shape[0], (list, tuple)):
                # input_shape[0] é um shape (lista/tupla de dimensões)
                input_dim = input_shape[0][-1]
            else:
                # Fallback - assumir que é direto
                input_dim = input_shape[-1]
        else:
            # input_shape é uma shape simples
            input_dim = input_shape[-1] if hasattr(input_shape, '__getitem__') else 317
            
        # Garantir que input_dim é um inteiro
        if hasattr(input_dim, 'value'):
            input_dim = input_dim.value
        elif not isinstance(input_dim, int):
            input_dim = 317  # Fallback para o número de features conhecido
            
        # Criar peso para transformação linear
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.channels),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.channels,),
                initializer='zeros',
                trainable=True
            )
        
        super().build(input_shape)
    
    def call(self, inputs, **kwargs):
        """
        Forward pass da camada GCN.
        
        Args:
            inputs: Lista [features, adjacency] ou [features, adjacency, batch_index]
        """
        # Extrair features e adjacência
        if len(inputs) == 2:
            features, adjacency = inputs
        elif len(inputs) == 3:
            features, adjacency, _ = inputs  # Ignorar batch_index por enquanto
        else:
            raise ValueError(f"Esperado 2 ou 3 inputs, recebido {len(inputs)}")
        
        # Converter para tensores densos se necessário
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        
        if isinstance(adjacency, tf.SparseTensor):
            adjacency = tf.sparse.to_dense(adjacency)
        adjacency = tf.convert_to_tensor(adjacency, dtype=tf.float32)
        
        # Transformação linear: X @ W
        output = tf.matmul(features, self.kernel)
        
        # Convolução: A @ (X @ W)
        output = tf.matmul(adjacency, output)
        
        # Adicionar bias se especificado
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        
        # Aplicar ativação
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def get_config(self):
        """Retorna configuração da camada para serialização."""
        config = {
            'channels': self.channels,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class CustomGlobalAvgPool(layers.Layer):
    """
    Pooling global customizado que funciona corretamente com DisjointLoader.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = False
    
    def call(self, inputs, **kwargs):
        """
        Forward pass do pooling global.
        
        Args:
            inputs: Tensor de features ou [features, batch_index]
        """
        if isinstance(inputs, list) and len(inputs) == 2:
            features, batch_index = inputs
            # Usar batch_index para pooling por grafo
            return tf.math.segment_mean(features, batch_index)
        else:
            # Pooling simples (média de todos os nós)
            features = inputs
            return tf.reduce_mean(features, axis=0, keepdims=True)
    
    def get_config(self):
        """Retorna configuração da camada."""
        return super().get_config()


def gcn_filter(adjacency):
    """
    Aplica a normalização GCN à matriz de adjacência.
    Implementa: D^(-1/2) @ (A + I) @ D^(-1/2)
    """
    # Adicionar self-loops (A + I)
    identity = tf.eye(tf.shape(adjacency)[0], dtype=adjacency.dtype)
    adjacency_with_loops = adjacency + identity
    
    # Calcular graus
    degree = tf.reduce_sum(adjacency_with_loops, axis=1)
    
    # D^(-1/2)
    degree_inv_sqrt = tf.pow(degree + 1e-8, -0.5)
    degree_inv_sqrt = tf.linalg.diag(degree_inv_sqrt)
    
    # Normalização: D^(-1/2) @ A @ D^(-1/2)
    normalized = tf.matmul(
        tf.matmul(degree_inv_sqrt, adjacency_with_loops),
        degree_inv_sqrt
    )
    
    return normalized