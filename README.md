#  AG News Embeddings — Dataset Numérico via Sentence Transformers

> **Projeto:** Construção de Dataset Numérico via Embeddings de Texto  
> **Disciplina:** Ciência de Dados — Doutorado  
> **Modelo:** `sentence-transformers/all-MiniLM-L6-v2`  
> **Dataset:** AG News (fancyzhx/ag_news — Hugging Face)

---

##  Descrição

Este projeto constrói um **dataset tabular numérico** a partir de textos de notícias,
convertendo cada notícia em um vetor de **384 dimensões** usando o modelo
`all-MiniLM-L6-v2` (Sentence Transformers).

O objetivo é produzir uma base pronta para análises de:
- Visualização e exploração (EDA)
- Agrupamento (clustering)
- Classificação supervisionada
- Outras análises de Ciência de Dados

---

##  Dataset Gerado

| Atributo | Valor |
|---|---|
| **Arquivo** | `agnews_embeddings_1000.csv` |
| **Amostras** | 1.000 (250 por classe) |
| **Colunas** | 392 (1 ID + 384 embeddings + 7 metadados) |
| **Classes** | World, Sports, Business, Sci/Tech |
| **Rótulos** | Sim (supervisionado) |
| **Idioma** | Inglês |

### Estrutura das colunas

| Coluna | Tipo | Descrição |
|---|---|---|
| `sample_id` | int | Identificador único da amostra |
| `f_000` a `f_383` | float | Embeddings MiniLM (384 dimensões) |
| `label` | int | Rótulo numérico (0=World, 1=Sports, 2=Business, 3=Sci/Tech) |
| `class_name` | str | Nome da categoria |
| `source` | str | Origem: AG News / Hugging Face |
| `license` | str | Condição de uso |
| `n_words` | int | Nº de palavras da notícia |
| `n_chars` | int | Nº de caracteres da notícia |
| `text_preview` | str | Trecho inicial do texto (80 chars) |

---

##  Estrutura do Repositório

```
.
├── trabalho_eda_texto_agnews_COLAB.ipynb  # Notebook principal (Google Colab)
├── trabalho_eda_texto_agnews.py           # Script Python equivalente
├── agnews_embeddings_1000.csv             # Dataset gerado (1.000 amostras)
├── gerar_notebook_colab.py                # Script gerador do .ipynb
├── descrição.txt                          # Enunciado oficial do trabalho
└── README.md                              # Este arquivo
```

---

##  Como Executar

### No Google Colab (recomendado)

1. Faça upload do arquivo `trabalho_eda_texto_agnews_COLAB.ipynb`
2. Ative GPU: `Ambiente de execução -> Alterar tipo -> T4 GPU`
3. Execute tudo: `Ctrl + F9`

### Localmente

```bash
# Instalar dependências
pip install datasets sentence-transformers scikit-learn plotly matplotlib pandas

# Rodar o script
python trabalho_eda_texto_agnews.py
```

> No Windows, execute com: `$env:PYTHONUTF8=1; python trabalho_eda_texto_agnews.py`

---

##  Pipeline de Processamento

```
AG News (Hugging Face)
        ↓
Amostragem estratificada (250/classe x 4 = 1.000 textos)
        ↓
Sentence Transformer: all-MiniLM-L6-v2
        ↓
Vetor de 384 dimensões por notícia
        ↓
DataFrame tabular (1.000 x 392)
        ↓
Exportação: agnews_embeddings_1000.csv
```

---

##  Análises Realizadas (EDA)

1. **Pergunta de investigação** — separabilidade das categorias via embeddings
2. **Unidade de análise** — 1 notícia = 1 vetor de 384 dimensões
3. **Features** — embeddings MiniLM zero-shot
4. **Qualidade** — ausentes, duplicatas, balanceamento, outliers (IsolationForest)
5. **Visualizações** — comprimento dos textos, norma L2, Parallel Coordinates
6. **PCA 2D** — projeção com variância explicada
7. **Hipótese** — Sports > Sci/Tech > World ≈ Business
8. **Classificação** — Regressão Logística: Acurácia 78.3%, F1 macro 0.78
9. **Limitações** — tamanho da amostra, PCA 2D, modelo sem fine-tuning
10. **Extensão** — Label Noise: queda de 7.5 pp com 10% de rótulos errados

---

##  Fontes e Licenciamento

| Recurso | Fonte | Licença |
|---|---|---|
| **AG News dataset** | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | Academic / Non-commercial |
| **MiniLM model** | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache 2.0 |
| **Sentence Transformers** | [sbert.net](https://www.sbert.net/) | Apache 2.0 |

>  Este dataset é de uso **exclusivamente acadêmico**. Não contém dados pessoais ou sensíveis.

---

##  Resultados

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| World | 0.77 | 0.77 | 0.77 |
| **Sports** | **0.93** | **0.90** | **0.92** |
| Business | 0.69 | 0.73 | 0.71 |
| Sci/Tech | 0.76 | 0.73 | 0.75 |
| **Macro avg** | **0.79** | **0.78** | **0.78** |

> Acurácia geral: **78.3%** com Regressão Logística (zero-shot, sem fine-tuning)
