# fiap_tech_challenge_04
Projeto que desenvolve um modelo LSTM para prever preços de ações. Inclui coleta de dados, criação e treinamento do modelo. Entregáveis: código-fonte, documentação e scripts.

## Project Structure

### General Structure

- **`main.py`**: Script principal que executa todo o pipeline do projeto.
- **`/src`**: contém todo o código-fonte do projeto, incluindo scripts para coleta de dados, pré-processamento e treinamento de modelo.
- **`run.py`**: Script para executar todo o pipeline do projeto a partir da raiz do repositório.

## Como processar

1. Instale todas as dependências necessárias:
   ```bash
   pip install -r requirements.txt
2. Configure os parametetros (`param`) em `src/ml/main.py` de acordo com o que precisar.
3. Execute a pipeline usando:
   ```bash
   python run.py