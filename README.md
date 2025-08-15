# ProjetoIA
## Descrição do Projeto

Este projeto visa replicar e estender o trabalho descrito no artigo "Emotion classification on eye-tracking and electroencephalograph fused signals employing deep gradient neural networks" (primeiro anexo), mas substituindo a arquitetura de rede neural por uma **Graph Neural Network (GNN)** para classificação de emoções multimodais. Usamos o dataset **SEED-V** (descrito no segundo artigo: "Comparing Recognition Performance and Robustness of Multimodal Deep Learning Models for Multimodal Emotion Recognition"), que inclui sinais de EEG e eye-tracking coletados durante a visualização de clipes de filmes que induzem 5 emoções: happy, sad, fear, disgust e neutral.

O foco inicial é gerar **grafos sacádicos** (mapas de movimentos oculares) para cada vídeo assistido por cada sujeito. Cada grafo representa o percurso óptico: 
- **Nós**: Pontos de fixação, com features como timestamps, duração da fixação, tamanho da pupila, dispersão e um placeholder para features EEG.
- **Arestas**: Sacadas (movimentos rápidos dos olhos), com duração e amplitude.

## Requisitos e Instalação

### Pré-requisitos
- Python 3.8+.
**Instale Dependências**:
pip install -r requirements.txt