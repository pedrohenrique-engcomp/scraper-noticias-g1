# pipeline_svg_embeddings_g1.py (versão ajustada sem Selenium, com heurística + prints)

import os
import re
import time
import textwrap
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ----------------- CONFIG -----------------
SECTIONS = {
    "politica": "https://g1.globo.com/politica/",
    "economia": "https://g1.globo.com/economia/",
    "tecnologia": "https://g1.globo.com/tecnologia/"
}

TARGET_PER_CLASS = 1000  # links por seção

# Caminho seguro para salvar arquivos no VS Code
OUTPUT_DIR = r".\output_pipeline"
RAW_CSV = os.path.join(OUTPUT_DIR, "g1_raw.csv")
PROCESSED_CSV = os.path.join(OUTPUT_DIR, "g1_processed.csv")
SVG_DIR = os.path.join(OUTPUT_DIR, "svgs")
EMBEDDINGS_NPY = os.path.join(OUTPUT_DIR, "embeddings.npy")
META_CSV = os.path.join(OUTPUT_DIR, "with_embeddings_meta.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
USE_PCA = True
PCA_COMPONENTS = 100

# Garantir que os diretórios existem
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SVG_DIR, exist_ok=True)

# ----------------- HELPERS -----------------
def sanitize_filename(s):
    s = re.sub(r'[^A-Za-z0-9_\- ]+', '', s)
    s = s.strip().replace(' ', '_')[:120]
    return s or "item"

# ----------------- SCRAPING (heurística pura) -----------------
def get_article_links_with_heuristic(section_url, max_links=1000, max_pages=500):
    """
    Coleta links de artigos do G1 usando heurística de paginação.
    Tenta diferentes padrões de paginação para evitar loop infinito.
    """

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"}
    links = set()

    for page in range(1, max_pages + 1):
        # possíveis padrões de paginação
        candidate_urls = [
            f"{section_url}index/feed/pagina-{page}.ghtml"
        ]

        found_new = False
        for url in candidate_urls:
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code != 200:
                    continue

                soup = BeautifulSoup(r.text, "html.parser")
                anchors = soup.find_all("a", href=True)
                before = len(links)

                for a in anchors:
                    href = a["href"]
                    if href.startswith("/"):
                        href = "https://g1.globo.com" + href
                    if href.endswith(".ghtml") and not ("index/feed/pagina-" in href):
                        links.add(href)
                        if len(links) >= max_links:
                            print(f"[OK] Alvo de {max_links} links atingido!")
                            return list(links)

                added = len(links) - before
                if added > 0:
                    found_new = True
                    print(f"[DEBUG] Página {page} ({url}) -> {added} novos links")
                    for l in list(links)[-added:]:
                        print("   ", l)
                    break  # já achou links com este padrão, não precisa testar os outros

            except Exception as e:
                print(f"[ERRO] {url}: {e}")

        if not found_new:
            print(f"[WARN] Página {page} não trouxe novos links (tentando próxima).")

        time.sleep(0.5)  # respeitar servidor

    print(f"[FIM] Total coletado de {section_url}: {len(links)} links")
    return list(links)


def fetch_article_text(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        article = soup.find("article")
        if not article:
            return ""
        return article.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def scrape_sections_g1(sections, target_per_class=TARGET_PER_CLASS):
    rows = []
    for label, url in sections.items():
        print(f"\n> Coletando seção {label} -> {url}")
        links = get_article_links_with_heuristic(url, max_links=target_per_class)
        print(f"  ✅ {len(links)} links encontrados em {label}")
        for link in tqdm(links, desc=f"Baixando artigos {label}"):
            text = fetch_article_text(link)
            if len(text) < 50:
                continue
            rows.append({
                "title": link.split("/")[-1],
                "text": text,
                "label": label,
                "url": link
            })
            time.sleep(0.2)
    return pd.DataFrame(rows)

# ----------------- SVG -----------------
def text_to_svg(text, embedding=None, title=None, filename='out.svg', width=800,
                font_size=14, line_height=1.2, max_chars_per_line=90):
    import svgwrite
    lines = []
    for paragraph in text.split('\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=max_chars_per_line)
        lines.extend(wrapped)
    est_height = int((len(lines) + 5) * font_size * line_height) + 40
    if est_height < 600:
        est_height = 600
    dwg = svgwrite.Drawing(filename, size=(f"{width}px", f"{est_height}px"))
    y = 20
    if title:
        dwg.add(dwg.text(title, insert=(10, y),
                         style=f"font-size:{font_size+2}px; font-family:Arial; font-weight:bold"))
        y += int(font_size * line_height * 1.5)
    for i, line in enumerate(lines):
        dy = y + i * font_size * line_height
        size = font_size
        if embedding is not None:
            size += np.linalg.norm(embedding) * 5
        dwg.add(dwg.text(line, insert=(10, dy), style=f"font-size:{size}px; font-family:Arial"))
    dwg.save()

# ----------------- EMBEDDINGS -----------------
def compute_embeddings(texts, model_name=MODEL_NAME, batch_size=32, show_progress=True):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True)
    embeddings = normalize(embeddings, norm='l2')
    if USE_PCA:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        embeddings = pca.fit_transform(embeddings)
    return embeddings

# ----------------- SPLIT -----------------
def split_save(embeddings, meta_df, output_dir=OUTPUT_DIR):
    labels = meta_df['label'].values
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_rest, y_train, y_rest, idx_train, idx_rest = train_test_split(
        embeddings, y, meta_df.index.values, test_size=0.3, stratify=y, random_state=42)

    val_frac_of_rest = 2/3
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_rest, y_rest, idx_rest, test_size=(1-val_frac_of_rest), stratify=y_rest, random_state=42)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)

    train_meta = meta_df.loc[idx_train].reset_index(drop=True)
    val_meta = meta_df.loc[idx_val].reset_index(drop=True)
    test_meta = meta_df.loc[idx_test].reset_index(drop=True)

    train_meta.to_csv(os.path.join(output_dir, "meta_train.csv"), index=False)
    val_meta.to_csv(os.path.join(output_dir, "meta_val.csv"), index=False)
    test_meta.to_csv(os.path.join(output_dir, "meta_test.csv"), index=False)

    print("✅ Divisões salvas: X_train.npy, X_val.npy, X_test.npy e meta_*.csv")
    return (X_train, X_val, X_test), (train_meta, val_meta, test_meta), le

# ----------------- MAIN -----------------
def main():
    df = scrape_sections_g1(SECTIONS, target_per_class=TARGET_PER_CLASS)
    df.to_csv(RAW_CSV, index=False)

    df['text'] = df['text'].astype(str).apply(lambda s: re.sub(r'\s+', ' ', s).strip())
    df['text_proc'] = df['text']
    df.to_csv(PROCESSED_CSV, index=False)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fname = sanitize_filename(f"{row['label']}_{idx}_{row['title'][:50]}")
        svg_path = os.path.join(SVG_DIR, f"{fname}.svg")
        try:
            text_to_svg(row['text'], title=row['title'], filename=svg_path)
            df.at[idx, 'svg_path'] = svg_path
        except:
            df.at[idx, 'svg_path'] = ""
    df.to_csv(PROCESSED_CSV, index=False)

    texts = df['text_proc'].astype(str).tolist()
    embeddings = compute_embeddings(texts)
    np.save(EMBEDDINGS_NPY, embeddings)
    df.to_csv(META_CSV, index=False)

    split_save(embeddings, df, output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()
