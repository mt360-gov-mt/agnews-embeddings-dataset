# -*- coding: utf-8 -*-
"""
=============================================================================
TRABALHO — EDA MULTIMODAL: MODALIDADE TEXTO
Dataset: AG News
Extrator: Sentence Transformer — all-MiniLM-L6-v2
Autor: [Seu Nome]
=============================================================================

Lógica central:
  Pergunta → Representação → Qualidade → Exploração → Hipótese → Classificação

Atividade baseada na Seção 10 do Laboratório de Ciência de Dados — EDA Multimodal.
"""

# =============================================================================
# SEÇÃO 0 — Setup do ambiente
# Execute esta célula primeiro no Google Colab
# =============================================================================

# !pip install -q datasets sentence-transformers plotly

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import IsolationForest
from pandas.plotting import parallel_coordinates

# Reprodutibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

pd.set_option("display.max_columns", 50)

print("✅ Ambiente configurado com sucesso.")


# =============================================================================
# SEÇÃO 1 — PERGUNTA DE INVESTIGAÇÃO
# =============================================================================

"""
ITEM 1: Pergunta de Investigação
---------------------------------
Os embeddings semânticos gerados pelo Sentence Transformer (MiniLM)
conseguem capturar diferenças temáticas entre notícias de forma que
um classificador simples — como a Regressão Logística — consiga
separar as quatro categorias do AG News (World, Sports, Business,
Sci/Tech) com desempenho razoável?

Especificamente: categorias com linguagem mais distinta (ex. Sports vs
Sci/Tech) serão mais separáveis do que categorias com vocabulário
mais próximo (ex. Business vs World)?
"""

print("=" * 65)
print("PERGUNTA DE INVESTIGAÇÃO")
print("=" * 65)
print("""
Os embeddings semânticos do MiniLM conseguem separar as
quatro categorias de notícias do AG News:

  → World    (Mundo / Política internacional)
  → Sports   (Esportes)
  → Business (Negócios / Economia)
  → Sci/Tech (Ciência e Tecnologia)

...de forma que um classificador simples (Regressão Logística)
alcance desempenho razoável sem fine-tuning do modelo?

Hipótese antecipada: categorias com vocabulário mais distinto
(Sports vs Sci/Tech) serão mais separáveis do que pares com
linguagem próxima (Business vs World).
""")

print("✅ Item 1 — Pergunta de investigação definida.")


# =============================================================================
# SEÇÃO 2 — UNIDADE DE ANÁLISE + CARREGAMENTO DOS DADOS
# =============================================================================

"""
ITEM 2: Unidade de Análise
---------------------------
Cada LINHA do DataFrame representa UMA NOTÍCIA do dataset AG News.
Após a geração dos embeddings, cada notícia é representada por um
vetor numérico de 384 dimensões (gerado pelo MiniLM).

  → Linha  = 1 notícia (título + corpo em inglês)
  → Coluna = 1 dimensão do embedding semântico (f_000 até f_383)
  → Classe = coluna 'class_name'
             World    → notícias de política e assuntos internacionais
             Sports   → notícias esportivas
             Business → notícias de negócios e economia
             Sci/Tech → notícias de ciência e tecnologia

O objetivo é verificar se as 384 dimensões do embedding semântico
são suficientes para distinguir os quatro temas automaticamente.
"""

from datasets import load_dataset

AG_LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
N_TEXT_PER_CLASS = 250   # amostras por classe → 250 × 4 classes = 1.000 amostras (mínimo exigido)

print("=" * 65)
print("ITEM 2 — UNIDADE DE ANÁLISE")
print("=" * 65)
print(f"""
Unidade de análise: 1 notícia do AG News

  Cada linha    → 1 notícia convertida em vetor de 384 dimensões
  Cada coluna   → 1 feature semântica do MiniLM (f_000 a f_383)
  Variável alvo → class_name ({' / '.join(AG_LABEL_NAMES)})
  Amostra       → {N_TEXT_PER_CLASS} notícias por classe
                  = {N_TEXT_PER_CLASS * len(AG_LABEL_NAMES)} notícias no total
""")

# -------------------------------------------------------------------
# Carregando o AG News do Hugging Face
# -------------------------------------------------------------------
print("Carregando dataset AG News...")
ag = load_dataset("fancyzhx/ag_news")

# Amostragem estratificada
text_items = []
for label_id in range(len(AG_LABEL_NAMES)):
    subset = ag["train"].filter(lambda x: x["label"] == label_id)
    subset = subset.shuffle(seed=RANDOM_STATE).select(range(N_TEXT_PER_CLASS))
    text_items.extend([subset[i] for i in range(len(subset))])

# DataFrame bruto
df_raw = pd.DataFrame({
    "text" : [x["text"]  for x in text_items],
    "label": [x["label"] for x in text_items],
})
df_raw["class_name"] = df_raw["label"].map(dict(enumerate(AG_LABEL_NAMES)))

print(f"\nDataset carregado: {df_raw.shape[0]} notícias × {df_raw.shape[1]} colunas\n")

# Distribuição das classes
print("Distribuição das classes:")
print(df_raw["class_name"].value_counts().to_string())

# Prévia dos textos
print("\nExemplos de notícias por categoria:")
print("-" * 65)
for cat in AG_LABEL_NAMES:
    exemplo = df_raw[df_raw["class_name"] == cat]["text"].iloc[0]
    print(f"\n[{cat}]\n{exemplo[:200]}...")
print("-" * 65)

print("\n✅ Item 2 — Unidade de análise concluído.")


# =============================================================================
# SEÇÃO 3 — FEATURES UTILIZADAS (Extração de Embeddings)
# =============================================================================

"""
ITEM 3: Features Utilizadas
-----------------------------
Usaremos o modelo 'all-MiniLM-L6-v2' da biblioteca Sentence Transformers
para converter cada notícia em um vetor de 384 dimensões.

Por que MiniLM?
  - Treinado especificamente para gerar embeddings semânticos de sentenças
  - Rápido e eficiente (versão destilada do BERT)
  - Captura similaridade semântica: textos com sentido parecido
    ficam próximos no espaço vetorial

Formato final: DataFrame com shape (480, 386)
  → 480 linhas (notícias) × 384 features + colunas 'label' e 'class_name'
"""

import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

print("=" * 65)
print("ITEM 3 — FEATURES UTILIZADAS")
print("=" * 65)
print("""
Modelo    : sentence-transformers/all-MiniLM-L6-v2
Dimensão  : 384 features por notícia
Estratégia: extração direta (zero-shot), sem fine-tuning
""")

# Carregando o modelo
print("Carregando Sentence Transformer...")
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# Gerando embeddings
print("Gerando embeddings — aguarde...")
X_text = text_model.encode(
    df_raw["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False
)

print(f"\nFormato dos embeddings: {X_text.shape}")
print(f"  → {X_text.shape[0]} notícias × {X_text.shape[1]} dimensões\n")

# Convertendo para DataFrame tabular
text_feature_names = [f"f_{i:03d}" for i in range(X_text.shape[1])]

df_text = pd.DataFrame(X_text, columns=text_feature_names)
df_text["label"]      = df_raw["label"].values
df_text["class_name"] = df_raw["class_name"].values

print("Prévia do DataFrame de embeddings (primeiras 5 linhas, primeiras 6 features):")
display_cols = text_feature_names[:6] + ["class_name"]
print(df_text[display_cols].head().to_string())

print(f"\nShape final: {df_text.shape}")
print("\n✅ Item 3 — Features extraídas com sucesso.")


# =============================================================================
# CONSTRUÇÃO DO DATASET FINAL (entregável obrigatório)
# =============================================================================
"""
Dataset final estruturado conforme os requisitos do enunciado:
  - sample_id     : identificador único da amostra
  - f_000 a f_383 : embeddings MiniLM (384 dimensões)
  - label         : rótulo numérico (0–3)
  - class_name    : nome da categoria
  - source        : origem do dado
  - license       : condição de uso
  - n_words       : metadado — nº de palavras da notícia
  - n_chars       : metadado — nº de caracteres da notícia
  - text_preview  : trecho inicial do texto (50 primeiros caracteres)
"""

# Metadados textuais
df_raw["n_chars"] = df_raw["text"].str.len()
df_raw["n_words"] = df_raw["text"].str.split().str.len()

# Montagem do dataset final
df_final = df_text.copy()
df_final.insert(0, "sample_id",    range(len(df_final)))   # identificador único
df_final["source"]       = "AG News — fancyzhx/ag_news (Hugging Face)"
df_final["license"]      = "Academic / Non-commercial use"
df_final["n_words"]      = df_raw["n_words"].values
df_final["n_chars"]      = df_raw["n_chars"].values
df_final["text_preview"] = df_raw["text"].str[:80].values

print("=" * 65)
print("DATASET FINAL — ESTRUTURA")
print("=" * 65)
print(f"\nShape : {df_final.shape[0]} amostras × {df_final.shape[1]} colunas")
print(f"  → {df_final.shape[0]} amostras {'✅ (≥ 1.000)' if df_final.shape[0] >= 1000 else '❌ (< 1.000)'}")
print("\nColunas:")
for col in df_final.columns[:6].tolist() + ["..."] + df_final.columns[-5:].tolist():
    print(f"  {col}")

print("\nPrévia (primeiras 3 linhas — colunas não-embedding):")
meta_cols = ["sample_id", "label", "class_name", "source", "license", "n_words", "n_chars"]
print(df_final[meta_cols].head(3).to_string())

# Exportação para CSV
csv_path = "agnews_embeddings_1000.csv"
df_final.to_csv(csv_path, index=False, encoding="utf-8")
print(f"\n✅ Dataset exportado: '{csv_path}'")
print(f"   Tamanho: {df_final.shape[0]} linhas × {df_final.shape[1]} colunas")
print(f"   Arquivo: {__import__('os').path.abspath(csv_path)}")


# =============================================================================
# SEÇÃO 4 — ANÁLISE DE QUALIDADE DOS EMBEDDINGS
# =============================================================================

"""
ITEM 4: Análise de Qualidade
------------------------------
Verificamos quatro tipos de problema nos embeddings gerados:

  1. Valores ausentes  → embeddings com NaN (problema no pipeline)
  2. Duplicatas        → vetores exatamente iguais (textos duplicados)
  3. Balanceamento     → distribuição igualitária das classes?
  4. Outliers          → IsolationForest para detecção multivariada

NOTA: NaN em embeddings quase sempre indica problema no pipeline
de processamento, não no fenômeno real.
"""

print("=" * 65)
print("ITEM 4 — ANÁLISE DE QUALIDADE DOS EMBEDDINGS")
print("=" * 65)

# --- 4.1 Valores Ausentes ---
print("\n[4.1] Valores ausentes nos embeddings:")
missing = df_text[text_feature_names].isna().sum()
total_missing = missing.sum()
if total_missing == 0:
    print("  ✅ Nenhum valor ausente encontrado.")
else:
    print(f"  ⚠️  {total_missing} valores ausentes detectados!")
    print(missing[missing > 0])

# --- 4.2 Duplicatas ---
print("\n[4.2] Duplicatas (vetores de embedding idênticos):")
n_dup = df_text[text_feature_names].duplicated().sum()
if n_dup == 0:
    print("  ✅ Nenhuma duplicata encontrada.")
else:
    print(f"  ⚠️  {n_dup} vetores duplicados detectados.")

# Duplicatas no texto bruto
n_dup_texto = df_raw["text"].duplicated().sum()
print(f"  Textos brutos duplicados: {n_dup_texto}")

# --- 4.3 Balanceamento das classes ---
print("\n[4.3] Balanceamento das classes:")
contagem = df_text["class_name"].value_counts()
print(contagem.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
contagem.plot(kind="bar", ax=ax, color=["#4e79a7","#f28e2b","#e15759","#76b7b2"],
              edgecolor="black")
ax.set_title("Distribuição das Classes — AG News (amostra)", fontweight="bold")
ax.set_xlabel("Categoria")
ax.set_ylabel("Nº de Notícias")
ax.set_xticklabels(contagem.index, rotation=20)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.show()
print("  → Dataset balanceado (amostragem estratificada de 120/classe)")

# --- 4.4 Outliers com Isolation Forest ---
print("\n[4.4] Outliers multivariados — Isolation Forest (contamination=5%):")
X_scaled_qlt = StandardScaler().fit_transform(X_text)

iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
pred_iso = iso.fit_predict(X_scaled_qlt)
scores    = iso.decision_function(X_scaled_qlt)

n_outliers = (pred_iso == -1).sum()
print(f"  Candidatos a outlier: {n_outliers} de {len(pred_iso)} notícias "
      f"({n_outliers/len(pred_iso)*100:.1f}%)")

# Mostrar os 5 outliers mais extremos
df_iso = df_raw.copy()
df_iso["anomaly_score"] = scores
df_iso["outlier"]       = pred_iso

print("\n  Top 5 outliers mais extremos (menor anomaly_score = mais anômalo):")
top_out = (df_iso[df_iso["outlier"] == -1]
           .sort_values("anomaly_score")
           .head(5)[["class_name", "anomaly_score", "text"]])
for _, row in top_out.iterrows():
    print(f"\n  [{row['class_name']}] score={row['anomaly_score']:.4f}")
    print(f"  {row['text'][:150]}...")

print("\n✅ Item 4 — Análise de qualidade concluída.")


# =============================================================================
# SEÇÃO 5 — VISUALIZAÇÕES (Univariada e Multivariada)
# =============================================================================

"""
ITEM 5: Visualizações
----------------------
5a. Univariada: comprimento dos textos por categoria
    → Verifica se categorias têm padrões distintos de tamanho

5b. Univariada: distribuição da norma L2 dos embeddings por classe
    → Verifica a "energia" vetorial — embeddings muito diferentes
      em magnitude podem indicar notícias atípicas

5c. Multivariada: Parallel Coordinates nos 8 primeiros componentes PCA
    → Mostra padrões multidimensionais por categoria
"""

print("=" * 65)
print("ITEM 5 — VISUALIZAÇÕES")
print("=" * 65)

# Padronização
X_text_scaled = StandardScaler().fit_transform(X_text)

# --- 5a. EDA do texto bruto: comprimento ---
print("\n[5a] Distribuição do comprimento dos textos por categoria:")
df_raw["n_chars"] = df_raw["text"].str.len()
df_raw["n_words"] = df_raw["text"].str.split().str.len()

print(df_raw.groupby("class_name")[["n_chars", "n_words"]].describe().T.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for class_name, grp in df_raw.groupby("class_name"):
    axes[0].hist(grp["n_chars"], bins=20, alpha=0.55, label=class_name)
    axes[1].hist(grp["n_words"], bins=20, alpha=0.55, label=class_name)

axes[0].set_title("Distribuição: nº de caracteres por categoria", fontweight="bold")
axes[0].set_xlabel("Nº de caracteres")
axes[0].set_ylabel("Frequência")
axes[0].legend()

axes[1].set_title("Distribuição: nº de palavras por categoria", fontweight="bold")
axes[1].set_xlabel("Nº de palavras")
axes[1].set_ylabel("Frequência")
axes[1].legend()

plt.tight_layout()
plt.show()

# --- 5b. Norma L2 dos embeddings por classe ---
print("\n[5b] Distribuição da norma L2 dos embeddings por categoria:")
normas = np.linalg.norm(X_text, axis=1)
df_normas = pd.DataFrame({"norma_l2": normas, "class_name": df_raw["class_name"].values})

fig, ax = plt.subplots(figsize=(9, 5))
cores = {"World": "#4e79a7", "Sports": "#f28e2b",
         "Business": "#e15759", "Sci/Tech": "#76b7b2"}

df_normas.boxplot(
    column="norma_l2", by="class_name",
    ax=ax, patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
    medianprops=dict(color="red", linewidth=2)
)
ax.set_title("Norma L2 dos embeddings por categoria", fontweight="bold")
ax.set_xlabel("Categoria")
ax.set_ylabel("Norma L2")
plt.suptitle("")
plt.tight_layout()
plt.show()
print("  → Normas semelhantes entre categorias indicam embeddings comparáveis em magnitude")

# --- 5c. Parallel Coordinates (8 PCs) ---
print("\n[5c] Parallel Coordinates — 8 primeiros componentes PCA:")
pca_vis = PCA(n_components=8, random_state=RANDOM_STATE)
X_pca8  = pca_vis.fit_transform(X_text_scaled)

df_pc8 = pd.DataFrame(X_pca8, columns=[f"PC{i+1}" for i in range(8)])
df_pc8["class_name"] = df_raw["class_name"].values

# Amostra de 12 por classe para não poluir o gráfico
parts = []
for cn, grp in df_pc8.groupby("class_name"):
    parts.append(grp.sample(min(len(grp), 12), random_state=RANDOM_STATE))
sample_pc8 = pd.concat(parts).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 5))
parallel_coordinates(
    sample_pc8, "class_name",
    color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"],
    alpha=0.45, linewidth=1.2, ax=ax
)
ax.set_title(
    "AG News + MiniLM — Parallel Coordinates (8 componentes PCA)",
    fontweight="bold", fontsize=13
)
ax.set_ylabel("Valor do componente")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

var_exp_8 = pca_vis.explained_variance_ratio_.sum() * 100
print(f"  → Variância explicada pelos 8 PCs: {var_exp_8:.1f}%")
print("\n✅ Item 5 — Visualizações concluídas.")


# =============================================================================
# SEÇÃO 6 — PROJEÇÃO PCA 2D
# =============================================================================

"""
ITEM 6: Projeção PCA 2D
------------------------
Reduzimos os embeddings de 384 → 2 dimensões para visualizar
graficamente a separação entre as categorias.

ATENÇÃO: PCA 2D preserva apenas uma fração da variância original.
Um espaço pouco separável em 2D pode ainda ser muito separável
em alta dimensão — por isso o PCA deve ser visto como ferramenta
de inspeção, não de conclusão.
"""

print("=" * 65)
print("ITEM 6 — PROJEÇÃO PCA 2D")
print("=" * 65)

pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d   = pca_2d.fit_transform(X_text_scaled)

var1 = pca_2d.explained_variance_ratio_[0] * 100
var2 = pca_2d.explained_variance_ratio_[1] * 100
var_total = var1 + var2

print(f"\nVariância explicada:")
print(f"  PC1: {var1:.2f}%")
print(f"  PC2: {var2:.2f}%")
print(f"  Total (2D): {var_total:.2f}%")
print(f"\n  → Os 2 componentes capturam apenas {var_total:.1f}% da variância.")
print(f"     Estrutura restante ({100-var_total:.1f}%) permanece oculta na projeção 2D.\n")

cores_cat = {
    "World"   : "#4e79a7",
    "Sports"  : "#f28e2b",
    "Business": "#e15759",
    "Sci/Tech": "#76b7b2"
}

fig, ax = plt.subplots(figsize=(9, 7))

for label_id, class_name in enumerate(AG_LABEL_NAMES):
    mask = df_raw["label"].values == label_id
    ax.scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        alpha=0.60, s=40,
        color=cores_cat[class_name],
        label=class_name, edgecolors="white", linewidths=0.3
    )

ax.set_xlabel(f"PC1 ({var1:.1f}% da variância)", fontsize=11)
ax.set_ylabel(f"PC2 ({var2:.1f}% da variância)", fontsize=11)
ax.set_title(
    f"AG News + MiniLM — PCA 2D\n"
    f"(variância total explicada: {var_total:.1f}%)",
    fontsize=13, fontweight="bold"
)
ax.legend(title="Categoria", fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

print("✅ Item 6 — PCA 2D concluído.")


# =============================================================================
# SEÇÃO 7 — HIPÓTESE SOBRE SEPARABILIDADE
# =============================================================================

"""
ITEM 7: Hipótese sobre Separabilidade
---------------------------------------
Com base nas visualizações anteriores (PCA 2D e Parallel Coordinates),
formulamos a seguinte hipótese ANTES de rodar o classificador:

HIPÓTESE:
  H1: Sports será a categoria mais fácil de separar das demais,
      pois seu vocabulário é altamente específico (nomes de atletas,
      esportes, competições) e visualmente aparece mais isolada no PCA.

  H2: Business e World terão maior sobreposição, pois compartilham
      termos de política econômica, geopolítica e mercados financeiros.

  H3: O modelo alcançará F1 macro ≥ 0.80, o que indicaria que os
      embeddings MiniLM são suficientemente discriminativos sem fine-tuning.

EXPECTATIVA por classe (do mais para o menos separável):
  Sports > Sci/Tech > World ≈ Business
"""

print("=" * 65)
print("ITEM 7 — HIPÓTESE SOBRE SEPARABILIDADE")
print("=" * 65)
print("""
Com base nas visualizações (PCA 2D + Parallel Coordinates):

H1: Sports → mais separável (vocabulário muito específico)
H2: Business e World → maior sobreposição (vocabulário compartilhado)
H3: F1 macro esperado ≥ 0.80 (embeddings MiniLM são robustos)

Ordem esperada de separabilidade:
  Sports > Sci/Tech > World ≈ Business
""")

print("✅ Item 7 — Hipótese formulada.")


# =============================================================================
# SEÇÃO 8 — CLASSIFICAÇÃO COM REGRESSÃO LOGÍSTICA
# =============================================================================

"""
ITEM 8: Teste com Regressão Logística
---------------------------------------
Usamos a Regressão Logística como classificador simples para testar
se as representações contêm informação discriminativa suficiente.

A complexidade baixa do classificador é intencional: se mesmo um modelo
linear consegue bom desempenho, as representações são de alta qualidade.

Split: 75% treino / 25% teste (estratificado por classe)
"""

print("=" * 65)
print("ITEM 8 — REGRESSÃO LOGÍSTICA")
print("=" * 65)

y_text = df_raw["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_text_scaled, y_text,
    test_size=0.25,
    stratify=y_text,
    random_state=RANDOM_STATE
)

print(f"\nSplit: {len(X_train)} treino / {len(X_test)} teste (75/25, estratificado)\n")

clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("─" * 65)
print("CLASSIFICATION REPORT")
print("─" * 65)
print(classification_report(y_test, y_pred, target_names=AG_LABEL_NAMES))

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=AG_LABEL_NAMES,
    xticks_rotation=30,
    ax=ax,
    colorbar=False
)
ax.set_title("AG News + MiniLM — Regressão Logística\nMatriz de Confusão",
             fontweight="bold")
plt.tight_layout()
plt.show()

# Comparação com a hipótese
from sklearn.metrics import f1_score, accuracy_score
f1_macro = f1_score(y_test, y_pred, average="macro")
acc      = accuracy_score(y_test, y_pred)

print(f"\nResultados:")
print(f"  Acurácia geral : {acc*100:.1f}%")
print(f"  F1 macro       : {f1_macro:.4f}")

print("\n─ Comparação com a Hipótese ─")
if f1_macro >= 0.80:
    print(f"  ✅ H3 CONFIRMADA: F1 macro = {f1_macro:.2f} ≥ 0.80")
else:
    print(f"  ❌ H3 NÃO CONFIRMADA: F1 macro = {f1_macro:.2f} < 0.80")

print("\n  Verificar H1 (Sports = melhor F1) e H2 (Business/World = mais confusão)")
print("  olhando o classification_report e a matriz de confusão acima.\n")

print("✅ Item 8 — Classificação concluída.")


# =============================================================================
# SEÇÃO 9 — LIMITAÇÕES DA ANÁLISE
# =============================================================================

"""
ITEM 9: Limitações da Análise
-------------------------------
Toda análise tem restrições. Reconhecê-las é parte do processo científico.
"""

print("=" * 65)
print("ITEM 9 — LIMITAÇÕES DA ANÁLISE")
print("=" * 65)
print("""
1. TAMANHO DA AMOSTRA
   Usamos apenas 120 notícias por classe (480 total) de um dataset
   com mais de 120.000 textos. Resultados podem não generalizar
   para o conjunto completo.

2. PCA COMO FERRAMENTA DE INSPEÇÃO
   A projeção 2D captura apenas uma fração da variância (ver Item 6).
   Classes que parecem sobrepostas no PCA 2D podem ser bem separáveis
   no espaço de 384 dimensões original.

3. MODELO SEM FINE-TUNING
   O MiniLM foi treinado para similaridade semântica geral, não
   especificamente para classificação de notícias. Fine-tuning
   no AG News provavelmente melhoraria os resultados.

4. OUTLIERS NÃO REMOVIDOS
   Identificamos outliers via IsolationForest mas não os removemos.
   Manter outliers é uma decisão — eles podem ser fenômenos
   legítimos ou ruído. Uma análise mais rigorosa investigaria
   cada caso individualmente.

5. CLASSIFICADOR LINEAR
   A Regressão Logística é intencionalmente simples. Modelos mais
   complexos (SVM, XGBoost, redes neurais) possivelmente alcançariam
   acurácia maior — mas o objetivo aqui é avaliar a representação,
   não o classificador.
""")

print("✅ Item 9 — Limitações documentadas.")


# =============================================================================
# SEÇÃO 10 — EXTENSÃO OPCIONAL: LABEL NOISE
# =============================================================================

"""
EXTENSÃO OPCIONAL: Impacto do Label Noise
------------------------------------------
Inserimos ruído nos rótulos de 10% das amostras de treino (trocando
a label por uma categoria aleatória diferente) e medimos o impacto
na acurácia da Regressão Logística.

Isso simula erros de rotulação que ocorrem em projetos reais.
"""

print("=" * 65)
print("EXTENSÃO — IMPACTO DO LABEL NOISE")
print("=" * 65)

# --- Treino com labels limpas (referência) ---
clf_clean = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
clf_clean.fit(X_train, y_train)
acc_clean = accuracy_score(y_test, clf_clean.predict(X_test))
f1_clean  = f1_score(y_test, clf_clean.predict(X_test), average="macro")

# --- Inserindo Label Noise em 10% do treino ---
NOISE_RATE = 0.10
y_train_noisy = y_train.copy()
n_noisy = int(len(y_train) * NOISE_RATE)

rng = np.random.default_rng(RANDOM_STATE)
noisy_idx = rng.choice(len(y_train), size=n_noisy, replace=False)

for idx in noisy_idx:
    label_original = y_train_noisy[idx]
    outras_classes = [l for l in range(len(AG_LABEL_NAMES)) if l != label_original]
    y_train_noisy[idx] = rng.choice(outras_classes)

print(f"\nNoise rate: {NOISE_RATE*100:.0f}% do treino "
      f"({n_noisy} de {len(y_train)} amostras com label trocada)\n")

# --- Treino com labels ruidosas ---
clf_noisy = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
clf_noisy.fit(X_train, y_train_noisy)
acc_noisy = accuracy_score(y_test, clf_noisy.predict(X_test))
f1_noisy  = f1_score(y_test, clf_noisy.predict(X_test), average="macro")

# --- Comparação ---
print("─" * 50)
print(f"{'Métrica':<20} {'Sem Ruído':>12} {'Com Ruído (10%)':>16}")
print("─" * 50)
print(f"{'Acurácia':<20} {acc_clean*100:>11.1f}% {acc_noisy*100:>15.1f}%")
print(f"{'F1 Macro':<20} {f1_clean:>12.4f} {f1_noisy:>16.4f}")
print("─" * 50)
print(f"\nImpacto do label noise:")
print(f"  Queda na acurácia : {(acc_clean - acc_noisy)*100:.1f} pontos percentuais")
print(f"  Queda no F1 macro : {(f1_clean - f1_noisy):.4f}\n")

# --- Matrizes de confusão lado a lado ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test, clf_clean.predict(X_test),
    display_labels=AG_LABEL_NAMES,
    xticks_rotation=30,
    ax=axes[0], colorbar=False
)
axes[0].set_title(f"Sem label noise\nAcc={acc_clean*100:.1f}%  F1={f1_clean:.2f}",
                  fontweight="bold")

ConfusionMatrixDisplay.from_predictions(
    y_test, clf_noisy.predict(X_test),
    display_labels=AG_LABEL_NAMES,
    xticks_rotation=30,
    ax=axes[1], colorbar=False
)
axes[1].set_title(f"Com label noise (10% treino)\nAcc={acc_noisy*100:.1f}%  F1={f1_noisy:.2f}",
                  fontweight="bold")

plt.suptitle("Impacto do Label Noise — AG News + MiniLM",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("✅ Extensão — Label Noise concluída.")

# =============================================================================
# FIM DO TRABALHO
# =============================================================================
print("\n" + "=" * 65)
print("TRABALHO CONCLUÍDO")
print("=" * 65)
print("""
Itens entregues:
  ✅ 1. Pergunta de investigação
  ✅ 2. Unidade de análise
  ✅ 3. Features utilizadas (MiniLM embeddings 384-dim)
  ✅ 4. Análise de qualidade (ausentes, duplicatas, outliers)
  ✅ 5. Visualizações (texto bruto, norma L2, Parallel Coords)
  ✅ 6. Projeção PCA 2D
  ✅ 7. Hipótese sobre separabilidade
  ✅ 8. Regressão Logística + métricas
  ✅ 9. Limitações da análise
  ✅ Extensão: Label Noise
""")
