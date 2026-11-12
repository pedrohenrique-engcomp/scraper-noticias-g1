# Coletor e Vetorizador de Notícias

Pipeline que coleta notícias de três editorias do portal G1, limpa o texto e entrega vetores prontos para classificação.

## Resultado

**5.537 artigos** coletados e processados, distribuídos entre Economia (1.941), Tecnologia (1.841) e Política (1.755), com divisão estratificada em treino (4.200), validação (1.199) e teste (601).

## O problema

Treinar um classificador de texto em português exige corpus rotulado. Em vez de depender de dataset pronto, o pipeline monta o próprio: a editoria do portal serve como rótulo natural, e a coleta gera um conjunto atual e do domínio desejado.

## Como funciona

1. **Coleta de links** — percorre a paginação de cada editoria, com heurística de parada quando a página deixa de render links novos.
2. **Extração do artigo** — baixa cada página e extrai título e corpo.
3. **Boas práticas de raspagem** — `User-Agent` identificado e intervalo entre requisições, para não sobrecarregar o servidor de origem.
4. **Vetorização** — embeddings com `all-MiniLM-L6-v2`, normalização L2 e redução a 100 dimensões por PCA.
5. **Divisão** — separação estratificada por editoria, preservando a proporção das classes.

## Stack

Python · requests · BeautifulSoup · Sentence-Transformers · scikit-learn · pandas · NumPy

## Como rodar

```bash
pip install -r requirements.txt
python pipeline_svg_embeddings.py
```

Os artefatos saem como `g1_raw.csv`, `g1_processed.csv`, `embeddings.npy` e os splits.

## Limitações

- Os seletores dependem da estrutura atual do portal; mudança de layout quebra a extração.
- A editoria é um rótulo aproximado: há matérias que transitam entre temas.
- A coleta é sequencial e demorada; não há paralelismo nem retomada de execução interrompida.
- Os dados coletados não são versionados aqui, por serem conteúdo de terceiros.

## Autoria

Pedro Henrique Acioly Amaral.
