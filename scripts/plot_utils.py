import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training_curves(history, model_name="Model", save_dir=None, y_true=None, y_pred=None, learning_rates=None):
    """
    Plota e salva curvas de acurácia, perda, gap de overfitting, learning rate e matriz de confusão.
    """
    if not hasattr(history, 'history'):
        print(f"[plot_training_curves] Nenhum histórico encontrado para {model_name}.")
        return
    hist = history.history
    plt.figure(figsize=(12, 5))
    """
    Plota e salva curvas de acurácia, perda, gap de overfitting, learning rate e matriz de confusão.
    """

    if not hasattr(history, 'history'):
        print(f"[plot_training_curves] Nenhum histórico encontrado para {model_name}.")
        return
    hist = history.history
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), '../data/training_plots')
    os.makedirs(save_dir, exist_ok=True)

    # 1. Curvas de acurácia e perda
    plt.figure(figsize=(12, 5))
    # Acurácia
    plt.subplot(1, 2, 1)
    if 'accuracy' in hist:
        plt.plot(hist['accuracy'], label='Treino')
    if 'val_accuracy' in hist:
        plt.plot(hist['val_accuracy'], label='Validação')
    plt.title(f'Acurácia - {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    # Perda
    plt.subplot(1, 2, 2)
    if 'loss' in hist:
        plt.plot(hist['loss'], label='Treino')
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='Validação')
    plt.title(f'Perda - {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.tight_layout()
    acc_loss_path = os.path.join(save_dir, f'{model_name.replace(" ", "_").lower()}_acc_loss.png')
    plt.savefig(acc_loss_path)
    plt.close()
    print(f"[plot_utils] Gráfico acurácia/perda salvo em: {acc_loss_path}")

    # 2. Gap de overfitting (acurácia treino - val)
    if 'accuracy' in hist and 'val_accuracy' in hist:
        plt.figure(figsize=(6, 4))
        gap = np.array(hist['accuracy']) - np.array(hist['val_accuracy'])
        plt.plot(gap, label='Gap (Treino - Validação)')
        plt.title(f'Gap de Overfitting - {model_name}')
        plt.xlabel('Época')
        plt.ylabel('Gap de Acurácia')
        plt.legend()
        gap_path = os.path.join(save_dir, f'{model_name.replace(" ", "_").lower()}_gap.png')
        plt.savefig(gap_path)
        plt.close()
        print(f"[plot_utils] Gráfico gap de overfitting salvo em: {gap_path}")

    # 3. Curva de learning rate (se disponível)
    if learning_rates is not None and len(learning_rates) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(learning_rates, label='Learning Rate')
        plt.title(f'Learning Rate - {model_name}')
        plt.xlabel('Época')
        plt.ylabel('Learning Rate')
        plt.legend()
        lr_path = os.path.join(save_dir, f'{model_name.replace(" ", "_").lower()}_lr.png')
        plt.savefig(lr_path)
        plt.close()
        print(f"[plot_utils] Gráfico learning rate salvo em: {lr_path}")

    # 4. Matriz de confusão (se y_true e y_pred fornecidos e não vazios)
    if (
        y_true is not None and y_pred is not None
        and hasattr(y_true, '__len__') and hasattr(y_pred, '__len__')
        and len(y_true) > 0 and len(y_pred) > 0
    ):
        plt.figure(figsize=(6, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Matriz de Confusão - {model_name}')
        cm_path = os.path.join(save_dir, f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"[plot_utils] Matriz de confusão salva em: {cm_path}")
    else:
        print(f"[plot_utils] Matriz de confusão NÃO gerada para {model_name} (sem dados válidos).");
