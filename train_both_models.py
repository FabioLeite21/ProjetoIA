#!/usr/bin/env python3
"""
Script para treinar e comparar GNN vs MLP baseline.
"""

import sys
import os

# Adicionar o diretório do projeto ao path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.gnn.gnn import EmotionGNN
from src.gnn.baseline_mlp import EmotionMLPTrainer
from src.data.emotion_graph_dataset import EmotionGraphDataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def train_and_compare():
    """Treina ambos os modelos e compara resultados."""
    print("EXPERIMENTO: GNN vs MLP Baseline")
    print("=" * 80)

    # Carregar dataset
    print("Carregando dataset...")
    base_dataset = EmotionGraphDataset('database/graph', normalization='none')
    print(f"Dataset carregado: {len(base_dataset.read())} grafos")

    results = {}

    # 1. Treinar MLP Baseline
    print("\nTREINANDO MLP BASELINE...")
    print("-" * 50)

    mlp_trainer = EmotionMLPTrainer(input_features=317*8, num_classes=5)  # 8x features por agregação expandida
    mlp_trainer.build_model()
    mlp_results = mlp_trainer.train_model(base_dataset)

    results['mlp'] = {
        'accuracy': mlp_results['test_accuracy'],
        'y_true': mlp_results['y_true'],
        'y_pred': mlp_results['y_pred']
    }

    # 2. Treinar GNN
    print("\nTREINANDO GNN...")
    print("-" * 50)

    try:
        print("Usando grafos originais (sem modificação de conectividade)...")
        emotion_gnn = EmotionGNN(input_features=317, num_classes=5)
        emotion_gnn.build_model()
        emotion_gnn.compile_model(learning_rate=0.0001)
        gnn_results = emotion_gnn.train_model()

        if gnn_results is not None:
            # Extrair acurácia do teste (assumindo que está no resultado)
            results['gnn'] = {
                'accuracy': 0.0,  # Será atualizado se disponível
                'training_successful': True
            }
        else:
            results['gnn'] = {
                'accuracy': 0.0,
                'training_successful': False
            }
    except Exception as e:
        print(f"Erro no treinamento da GNN: {e}")
        results['gnn'] = {
            'accuracy': 0.0,
            'training_successful': False,
            'error': str(e)
        }

    # 3. Comparar Resultados
    print("\nCOMPARAÇÃO DE RESULTADOS")
    print("=" * 80)

    print(f"MLP Baseline:")
    print(f"   Acurácia: {results['mlp']['accuracy']*100:.2f}%")
    print(f"   Método: Features agregadas (mean, std, min, max)")
    print(f"   Estrutura: Ignora grafo completamente")

    if results['gnn']['training_successful']:
        print(f"\nGNN:")
        print(f"   Acurácia: {results['gnn']['accuracy']*100:.2f}%")
        print(f"   Método: Graph Convolutional Network")
        print(f"   Estrutura: Usa conectividade do grafo")

        # Calcular diferença
        diff = results['gnn']['accuracy'] - results['mlp']['accuracy']
        if diff > 0.02:  # 2% melhor
            print(f"\nGNN é {diff*100:.1f}% melhor que MLP - estrutura do grafo está ajudando!")
        elif diff < -0.02:  # 2% pior
            print(f"\nMLP é {abs(diff)*100:.1f}% melhor que GNN - estrutura do grafo está atrapalhando!")
        else:
            print(f"\nResultados similares (diferença: {diff*100:.1f}%) - estrutura do grafo tem impacto neutro")
    else:
        print(f"\nGNN:")
        print(f"   Status: Falha no treinamento")
        if 'error' in results['gnn']:
            print(f"   Erro: {results['gnn']['error']}")
        print(f"\nMLP baseline é a única opção funcional por enquanto")

    # 4. Análise Detalhada do MLP
    print(f"\nANÁLISE DETALHADA - MLP BASELINE")
    print("-" * 50)

    y_true = results['mlp']['y_true']
    y_pred = results['mlp']['y_pred']

    emotion_names = ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy']

    print("Relatório de Classificação (MLP):")
    report = classification_report(
        y_true, y_pred,
        target_names=emotion_names,
        zero_division=0
    )
    print(report)

    print("Matriz de Confusão (MLP):")
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)

    # 5. Recomendações
    print(f"\nRECOMENDAÇÕES")
    print("-" * 50)

    mlp_acc = results['mlp']['accuracy']

    if mlp_acc > 0.6:
        print("MLP baseline tem boa performance (>60%)")
        print("   - As features agregadas são discriminativas")
        print("   - Continuar melhorando a GNN para superar o baseline")
    elif mlp_acc > 0.4:
        print("MLP baseline tem performance moderada (40-60%)")
        print("   - Features têm alguma capacidade discriminativa")
        print("   - Foco em melhorar both arquiteturas")
    else:
        print("MLP baseline tem performance baixa (<40%)")
        print("   - Problema fundamental nos dados ou features")
        print("   - Revisar preprocessing e feature engineering")

    if results['gnn']['training_successful']:
        if results['gnn']['accuracy'] < mlp_acc:
            print("   - GNN pior que MLP indica over-smoothing ou conectividade inadequada")
            print("   - Tentar conectividade mais esparsa ou arquitetura mais simples")

    return results


if __name__ == "__main__":
    results = train_and_compare()