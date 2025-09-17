"""
Utilitários para geração de gráficos e visualizações de treinamento.
Autor: Davi Augusto

Este módulo contém funções para visualizar:
- Loss e accuracy durante o treinamento
- Precisão, recall e F1-score por classe
- Learning rate ao longo das épocas
- Matriz de confusão
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime

# Configuração global do estilo dos gráficos - Davi Augusto
plt.style.use('default')
sns.set_palette("husl")

# Configurações de fonte e estilo
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})


def create_output_dir(base_dir="results"):
    """
    Cria diretório para salvar os gráficos com timestamp.
    
    Args:
        base_dir (str): Diretório base para salvar os resultados
        
    Returns:
        str: Caminho do diretório criado
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"training_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_training_history(history, output_dir):
    """
    Gera gráficos do histórico de treinamento com design moderno.
    Autor: Davi Augusto
    """
    # Configurar figura com subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Evolução da Loss e Accuracy Durante o Treinamento', fontsize=16, fontweight='bold', y=0.98)
    
    # Cores modernas
    train_color = '#2E86AB'  # Azul
    val_color = '#F24236'    # Vermelho
    
    # Gráfico de Loss
    epochs = range(1, len(history.history['loss']) + 1)
    ax1.plot(epochs, history.history['loss'], color=train_color, linewidth=2.5, label='Loss de Treino', marker='o', markersize=3)
    if 'val_loss' in history.history:
        ax1.plot(epochs, history.history['val_loss'], color=val_color, linewidth=2.5, label='Loss de Validação', marker='s', markersize=3)
    
    ax1.set_title('Evolução da Loss Durante o Treinamento', fontweight='bold', pad=20)
    ax1.set_xlabel('Época', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de Accuracy
    ax2.plot(epochs, history.history['accuracy'], color=train_color, linewidth=2.5, label='Accuracy de Treino', marker='o', markersize=3)
    if 'val_accuracy' in history.history:
        ax2.plot(epochs, history.history['val_accuracy'], color=val_color, linewidth=2.5, label='Accuracy de Validação', marker='s', markersize=3)
    
    ax2.set_title('Evolução da Accuracy Durante o Treinamento', fontweight='bold', pad=20)
    ax2.set_xlabel('Época', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Gráfico separado para Learning Rate se disponível
    if 'lr' in history.history:
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, history.history['lr'], color='#A23B72', linewidth=2.5, marker='o', markersize=4)
        plt.title('Evolução do Learning Rate', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Época', fontweight='bold')
        plt.ylabel('Learning Rate', fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    Gera matriz de confusão com design moderno e informativo.
    Autor: Davi Augusto
    """
    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Configurar figura com dois subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Análise de Confusão - Classificação de Emoções', fontsize=16, fontweight='bold', y=0.98)
    
    # Matriz de confusão - valores absolutos
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Matriz de Confusão - Contagens Absolutas', fontweight='bold', pad=20)
    ax1.set_xlabel('Predição', fontweight='bold')
    ax1.set_ylabel('Verdadeiro', fontweight='bold')
    
    # Matriz de confusão - percentuais
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Matriz de Confusão - Percentuais', fontweight='bold', pad=20)
    ax2.set_xlabel('Predição', fontweight='bold')
    ax2.set_ylabel('Verdadeiro', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_classification_metrics(detailed_metrics, class_names, output_dir):
    """
    Gera gráfico detalhado de métricas de classificação com design moderno.
    Autor: Davi Augusto
    """
    # Configurar figura com subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Layout: 2x2 com o último subplot ocupando espaço extra para o relatório
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])  # Precision
    ax2 = fig.add_subplot(gs[0, 1])  # Recall  
    ax3 = fig.add_subplot(gs[1, 0])  # F1-Score
    ax4 = fig.add_subplot(gs[:, 2])  # Relatório de texto
    
    fig.suptitle('Relatório Detalhado de Classificação', fontsize=18, fontweight='bold', y=0.95)
    
    # Cores para cada classe
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Precision por classe
    precision_values = detailed_metrics['precision_per_class']
    bars1 = ax1.bar(class_names, precision_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax1.set_title('Precisão por Classe', fontweight='bold', pad=15)
    ax1.set_ylabel('Precisão', fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, value in zip(bars1, precision_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Recall por classe
    recall_values = detailed_metrics['recall_per_class']
    bars2 = ax2.bar(class_names, recall_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax2.set_title('Recall por Classe', fontweight='bold', pad=15)
    ax2.set_ylabel('Recall', fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, value in zip(bars2, recall_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # F1-Score por classe
    f1_values = detailed_metrics['f1_per_class']
    bars3 = ax3.bar(class_names, f1_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax3.set_title('F1-Score por Classe', fontweight='bold', pad=15)
    ax3.set_ylabel('F1-Score', fontweight='bold')
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, value in zip(bars3, f1_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Relatório de texto
    ax4.axis('off')
    
    # Calcular total de amostras
    total_samples = len(detailed_metrics.get('test_predictions', []))
    if total_samples == 0:
        # Fallback: usar matriz de confusão se disponível
        if 'confusion_matrix' in detailed_metrics:
            total_samples = detailed_metrics['confusion_matrix'].sum()
    
    # Encontrar melhor e pior classe
    best_class_idx = np.argmax(f1_values)
    worst_class_idx = np.argmin(f1_values)
    
    report_text = f"""
📊 RELATÓRIO DE CLASSIFICAÇÃO

🎯 Métricas Globais:
• Accuracy: {detailed_metrics['accuracy']:.3f} ({detailed_metrics['accuracy']*100:.1f}%)

📈 Médias Macro:
• Precision: {detailed_metrics['precision_macro']:.3f}
• Recall: {detailed_metrics['recall_macro']:.3f}
• F1-Score: {detailed_metrics['f1_macro']:.3f}

⚖️ Médias Ponderadas:
• Precision: {detailed_metrics['precision_micro']:.3f}
• Recall: {detailed_metrics['recall_micro']:.3f}
• F1-Score: {detailed_metrics['f1_micro']:.3f}

🏆 Melhor Classe (F1):
• {class_names[best_class_idx]}: {f1_values[best_class_idx]:.3f}

⚠️ Pior Classe (F1):
• {class_names[worst_class_idx]}: {f1_values[worst_class_idx]:.3f}

📋 Total de Amostras: {total_samples}
"""
    
    ax4.text(0.05, 0.95, report_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_metrics.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_model_comparison(gnn_results, mlp_results, output_dir):
    """
    Gera gráfico comparativo entre modelos GNN e MLP com design moderno.
    Autor: Davi Augusto
    """
    # Configurar figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparação de Desempenho: GNN vs MLP', fontsize=18, fontweight='bold', y=0.95)
    
    # Cores dos modelos
    gnn_color = '#FF6B6B'
    mlp_color = '#4ECDC4'
    
    # 1. Comparação de Loss
    epochs_gnn = range(1, len(gnn_results['train_losses']) + 1)
    epochs_mlp = range(1, len(mlp_results['train_losses']) + 1)
    
    axes[0, 0].plot(epochs_gnn, gnn_results['train_losses'], color=gnn_color, linewidth=2.5, 
                    label='GNN Train', marker='o', markersize=4, alpha=0.8)
    axes[0, 0].plot(epochs_gnn, gnn_results['val_losses'], color=gnn_color, linewidth=2.5, 
                    linestyle='--', label='GNN Val', marker='s', markersize=4, alpha=0.8)
    axes[0, 0].plot(epochs_mlp, mlp_results['train_losses'], color=mlp_color, linewidth=2.5, 
                    label='MLP Train', marker='o', markersize=4, alpha=0.8)
    axes[0, 0].plot(epochs_mlp, mlp_results['val_losses'], color=mlp_color, linewidth=2.5, 
                    linestyle='--', label='MLP Val', marker='s', markersize=4, alpha=0.8)
    
    axes[0, 0].set_title('Evolução do Loss', fontweight='bold', pad=15)
    axes[0, 0].set_xlabel('Épocas', fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontweight='bold')
    axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Comparação de Accuracy
    axes[0, 1].plot(epochs_gnn, gnn_results['train_accuracies'], color=gnn_color, linewidth=2.5, 
                    label='GNN Train', marker='o', markersize=4, alpha=0.8)
    axes[0, 1].plot(epochs_gnn, gnn_results['val_accuracies'], color=gnn_color, linewidth=2.5, 
                    linestyle='--', label='GNN Val', marker='s', markersize=4, alpha=0.8)
    axes[0, 1].plot(epochs_mlp, mlp_results['train_accuracies'], color=mlp_color, linewidth=2.5, 
                    label='MLP Train', marker='o', markersize=4, alpha=0.8)
    axes[0, 1].plot(epochs_mlp, mlp_results['val_accuracies'], color=mlp_color, linewidth=2.5, 
                    linestyle='--', label='MLP Val', marker='s', markersize=4, alpha=0.8)
    
    axes[0, 1].set_title('Evolução da Accuracy', fontweight='bold', pad=15)
    axes[0, 1].set_xlabel('Épocas', fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Comparação de métricas finais
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    gnn_metrics = [
        gnn_results['detailed_metrics']['accuracy'],
        gnn_results['detailed_metrics']['precision_macro'],
        gnn_results['detailed_metrics']['recall_macro'],
        gnn_results['detailed_metrics']['f1_macro']
    ]
    mlp_metrics = [
        mlp_results['detailed_metrics']['accuracy'],
        mlp_results['detailed_metrics']['precision_macro'],
        mlp_results['detailed_metrics']['recall_macro'],
        mlp_results['detailed_metrics']['f1_macro']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = axes[1, 0].bar(x - width/2, gnn_metrics, width, label='GNN', 
                          color=gnn_color, alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = axes[1, 0].bar(x + width/2, mlp_metrics, width, label='MLP', 
                          color=mlp_color, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    axes[1, 0].set_title('Métricas Finais de Teste', fontweight='bold', pad=15)
    axes[1, 0].set_ylabel('Score', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics_names)
    axes[1, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim(0, 1.0)
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Resumo comparativo
    axes[1, 1].axis('off')
    
    # Calcular diferenças
    acc_diff = gnn_metrics[0] - mlp_metrics[0]
    f1_diff = gnn_metrics[3] - mlp_metrics[3]
    
    # Determinar modelo vencedor
    if acc_diff > 0:
        winner = "GNN"
        winner_color = gnn_color
    else:
        winner = "MLP"
        winner_color = mlp_color
    
    summary_text = f"""
🏆 RESUMO COMPARATIVO

📊 Modelo Vencedor: {winner}

🎯 Accuracy:
• GNN: {gnn_metrics[0]:.3f} ({gnn_metrics[0]*100:.1f}%)
• MLP: {mlp_metrics[0]:.3f} ({mlp_metrics[0]*100:.1f}%)
• Diferença: {abs(acc_diff):.3f}

📈 F1-Score Macro:
• GNN: {gnn_metrics[3]:.3f}
• MLP: {mlp_metrics[3]:.3f}
• Diferença: {abs(f1_diff):.3f}

⚡ Épocas de Treinamento:
• GNN: {len(gnn_results['train_losses'])} épocas
• MLP: {len(mlp_results['train_losses'])} épocas

🔍 Observações:
• Loss final GNN: {gnn_results['val_losses'][-1]:.4f}
• Loss final MLP: {mlp_results['val_losses'][-1]:.4f}
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=winner_color, alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_dataset_distribution(class_counts, class_names, output_dir):
    """
    Gera gráfico de distribuição do dataset com design moderno.
    Autor: Davi Augusto
    """
    # Configurar figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Distribuição do Dataset por Classes', fontsize=18, fontweight='bold', y=0.95)
    
    # Cores vibrantes para cada classe
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # 1. Gráfico de barras
    bars = ax1.bar(class_names, class_counts, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=2)
    
    ax1.set_title('Contagem por Classe', fontweight='bold', pad=20)
    ax1.set_ylabel('Número de Amostras', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(class_counts)*0.01,
                f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Rotacionar labels se necessário
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Gráfico de pizza
    wedges, texts, autotexts = ax2.pie(class_counts, labels=class_names, colors=colors, 
                                      autopct='%1.1f%%', startangle=90, 
                                      explode=[0.05]*len(class_names),
                                      shadow=True, textprops={'fontweight': 'bold'})
    
    ax2.set_title('Distribuição Percentual', fontweight='bold', pad=20)
    
    # Melhorar aparência do texto
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    # Adicionar estatísticas
    total_samples = sum(class_counts)
    mean_samples = np.mean(class_counts)
    std_samples = np.std(class_counts)
    
    # Calcular balanceamento
    max_count = max(class_counts)
    min_count = min(class_counts)
    balance_ratio = min_count / max_count
    
    stats_text = f"""
📊 ESTATÍSTICAS DO DATASET

📈 Total de Amostras: {total_samples:,}
📊 Média por Classe: {mean_samples:.1f}
📏 Desvio Padrão: {std_samples:.1f}

⚖️ Balanceamento:
• Maior classe: {max_count} amostras
• Menor classe: {min_count} amostras
• Razão de balanceamento: {balance_ratio:.2f}

{'✅ Dataset bem balanceado' if balance_ratio > 0.8 else '⚠️ Dataset desbalanceado'}
"""
    
    # Adicionar caixa de texto com estatísticas
    fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Espaço para as estatísticas
    plt.savefig(os.path.join(output_dir, 'dataset_distribution.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_summary_report(results_dict, save_dir=None):
    """
    Gera relatório resumo em texto.
    
    Args:
        results_dict (dict): Dicionário com resultados dos modelos
        save_dir (str): Diretório para salvar o relatório
    """
    if not results_dict:
        return
    
    report = []
    report.append("=" * 60)
    report.append("RELATÓRIO DE ANÁLISE DE TREINAMENTO")
    report.append("Autor: Davi Augusto")
    report.append("=" * 60)
    report.append(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    report.append("")
    
    for model_name, metrics in results_dict.items():
        report.append(f"MODELO: {model_name}")
        report.append("-" * 30)
        for metric, value in metrics.items():
            if isinstance(value, float):
                report.append(f"{metric.capitalize()}: {value:.4f}")
            else:
                report.append(f"{metric.capitalize()}: {value}")
        report.append("")
    
    # Encontrar melhor modelo
    if len(results_dict) > 1:
        best_model = max(results_dict.keys(), 
                        key=lambda x: results_dict[x].get('accuracy', 0))
        report.append(f"MELHOR MODELO: {best_model}")
        report.append(f"Accuracy: {results_dict[best_model].get('accuracy', 0):.4f}")
        report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    print(report_text)
    
    if save_dir:
        with open(os.path.join(save_dir, 'training_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Relatório salvo em: {save_dir}")