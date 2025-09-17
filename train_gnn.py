#!/usr/bin/env python3
"""
Script para treinar o modelo GNN de classificação de emoções.
"""

import sys
import os

# Adicionar o diretório do projeto ao path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.gnn.gnn import EmotionGNN

def main():
    """Função principal para treinar o modelo GNN."""
    print("Iniciando treinamento do modelo GNN para classificação de emoções...")
    print("=" * 60)
    
    # Criar instância do modelo
    emotion_gnn = EmotionGNN(input_features=317, num_classes=5)
    
    # Construir o modelo
    print("Construindo modelo...")
    emotion_gnn.build_model()
    
    # Compilar o modelo
    print("Compilando modelo...")
    emotion_gnn.compile_model(learning_rate=0.001)
    
    # Treinar o modelo
    print("Iniciando treinamento...")
    results = emotion_gnn.train_model()
    
    if results is not None:
        print("\nTreinamento concluído com sucesso!")
        
        # Salvar o modelo treinado
        model_path = os.path.join(project_root, "models", "emotion_gnn_trained.h5")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        emotion_gnn.save_model(model_path)
        
        print(f"Modelo salvo em: {model_path}")
    else:
        print("\nTreinamento falhou. Verifique os erros acima.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)