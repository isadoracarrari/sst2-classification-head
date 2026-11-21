# Trocando a LM Head por uma cabeça de classificação (SST-2)

Projeto da disciplina de Customizando LLMs (UFPR). Em vez de pedir ao LLM que responda "positivo"/"negativo" via prompt, a LM Head (um neurônio por token do vocabulário) é substituída por uma camada de classificação com **2 neurônios**, usando `AutoModelForSequenceClassification`.

Ideia baseada em Howard & Ruder (2018), *Universal Language Model Fine-tuning for Text Classification* (arXiv:1801.06146).

## Experimentos
Dataset: SST-2 (1000 exemplos de treino, validação completa como teste).

| Backbone | Modo | Parâmetros treináveis | Tempo | Acurácia val. |
|---|---|---|---|---|
| distilbert-base-uncased | fine-tuning completo | 66.955.010 | 20,0 s | 84,3% |
| distilbert-base-uncased | só a nova camada | 592.130 | 9,0 s | 50,9% |

## Como rodar
```bash
pip install -r requirements.txt
python classify.py --model distilbert-base-uncased
python classify.py --model distilbert-base-uncased --freeze-backbone
# Llama (modelo restrito): export HF_TOKEN=... e use --model meta-llama/Llama-3.2-1B
```
Os resultados ficam em `results/*.json`.

## Discussão
Foram comparados dois modos de treino com o backbone distilbert-base-uncased no SST-2 (1000 exemplos de treino, 3 épocas, avaliação no conjunto de validação).

No fine-tuning completo, todos os 66.955.010 parâmetros foram treinados e a acurácia de validação chegou a 84,3%, em 20 s de treino. Nas duas primeiras épocas a acurácia ficou em torno de 51%, e só na terceira o modelo passou a acertar com frequência.

No transfer learning (backbone congelado), apenas a nova camada de classificação foi treinada: 592.130 parâmetros, cerca de 0,9% do total. O treino foi mais rápido (9 s), mas a acurácia ficou em 50,9%, equivalente a um chute em um problema binário, e a loss quase não diminuiu.

Uma possível explicação é que a taxa de aprendizado (2e-5) é adequada para ajustar um modelo já treinado, mas pequena para uma camada inicializada do zero, que precisaria de mais épocas ou de uma taxa maior. Essa hipótese não foi testada aqui. Também é possível que a representação do DistilBERT sem ajuste não separe bem as classes com uma camada tão simples.

Limitações: cada configuração foi executada uma única vez, com poucos dados e poucas épocas, então a diferença entre os modos pode mudar com outros hiperparâmetros ou outras sementes.
