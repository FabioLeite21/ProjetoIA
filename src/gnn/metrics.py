"""
Métricas customizadas para avaliação do modelo GNN de classificação de emoções.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf


def evaluate_model_metrics(model, test_loader, num_classes=5):
    """
    Avalia o modelo usando métricas detalhadas de classification.
    
    Args:
        model: Modelo treinado
        test_loader: DisjointLoader com dados de teste
        num_classes: Número de classes (padrão: 5)
    
    Returns:
        dict: Dicionário com métricas detalhadas
    """
    print("Calculando métricas detalhadas no conjunto de teste...")
    
    # Coletar predições e targets
    all_predictions = []
    all_targets = []
    
    # Reinicializar o loader
    test_loader = test_loader.__class__(test_loader.dataset, batch_size=test_loader.batch_size, epochs=1)
    
    for batch in test_loader:
        inputs, targets = batch
        predictions = model(inputs, training=False)
        
        # Converter para numpy
        pred_classes = tf.argmax(predictions, axis=1).numpy()
        target_classes = targets.numpy() if hasattr(targets, 'numpy') else targets
        
        all_predictions.extend(pred_classes.tolist())
        all_targets.extend(target_classes.tolist())
    
    # Verificar se temos dados
    if len(all_predictions) == 0 or len(all_targets) == 0:
        print("Aviso: Nenhuma predição foi coletada. Retornando métricas vazias.")
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'message': 'Sem dados para avaliação'
        }
    
    # Converter para arrays numpy e garantir que sejam 1D
    y_true = np.array(all_targets).flatten()
    y_pred = np.array(all_predictions).flatten()
    
    print(f"Dados coletados: {len(y_true)} amostras")
    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    print(f"Classes únicas em y_true: {np.unique(y_true)}")
    print(f"Classes únicas em y_pred: {np.unique(y_pred)}")
    
    # Calcular métricas
    accuracy = np.mean(y_true == y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Métricas por classe
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Relatório de classificação
    class_names = ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy']
    
    # Verificar quais classes estão presentes
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    print(f"Classes em y_true: {unique_true}")
    print(f"Classes em y_pred: {unique_pred}")
    
    labels = list(range(num_classes))
    
    class_report = classification_report(
        y_true, y_pred, 
        labels=labels,
        target_names=class_names[:num_classes],
        zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    
    return metrics


def print_metrics_report(metrics, class_names=None):
    """
    Imprime um relatório detalhado das métricas.
    
    Args:
        metrics: Dicionário de métricas retornado por evaluate_model_metrics
        class_names: Lista de nomes das classes
    """
    if class_names is None:
        class_names = ['Disgust', 'Fear', 'Sad', 'Neutral', 'Happy']
    
    print("\n" + "="*60)
    print("RELATÓRIO DE MÉTRICAS - CLASSIFICAÇÃO DE EMOÇÕES")
    print("="*60)
    
    print(f"\nMÉTRICAS GERAIS:")
    print(f"Acurácia: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (Micro): {metrics['precision_micro']:.4f}")
    print(f"Recall (Micro): {metrics['recall_micro']:.4f}")
    print(f"F1-Score (Micro): {metrics['f1_micro']:.4f}")
    
    print(f"\nMÉTRICAS POR CLASSE:")
    for i, class_name in enumerate(class_names):
        if i < len(metrics['precision_per_class']):
            print(f"{class_name}:")
            print(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
            print(f"  Recall: {metrics['recall_per_class'][i]:.4f}")
            print(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}")
    
    print(f"\nMATRIZ DE CONFUSÃO:")
    print(metrics['confusion_matrix'])
    
    print(f"\nRELATÓRIO DE CLASSIFICAÇÃO:")
    print(metrics['classification_report'])
    print("="*60)