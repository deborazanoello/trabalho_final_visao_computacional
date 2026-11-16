"""Medidas simples de cada imagem, imagens repetidas e o teste de atalho."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from .comum import Pastas, caminho_absoluto

MEDIDAS = ["largura", "altura", "aspecto", "intensidade_media", "contraste", "tinta", "componentes", "fundo_escuro"]

# Nomes usados nas tabelas e nos gráficos.
NOMES = {
    "largura": "largura (px)",
    "altura": "altura (px)",
    "aspecto": "largura / altura",
    "intensidade_media": "intensidade média",
    "contraste": "contraste (desvio)",
    "tinta": "proporção de tinta",
    "componentes": "componentes conexos",
    "fundo_escuro": "fundo escuro",
    "riscada": "riscada",
    "riscada_ilegivel": "riscada ilegível",
    "correcao": "correção",
    "ilegivel": "ilegível",
    "imagens": "imagens",
    "arquivos_faltando": "arquivos faltando",
    "grupos_repetidos": "grupos repetidos",
    "imagens_em_grupos_repetidos": "imagens nesses grupos",
    "grupos_em_mais_de_um_split": "grupos em mais de uma partição",
    "grupos_com_rotulos_diferentes": "grupos com rótulos diferentes",
}


def ler_cinza(caminho) -> np.ndarray | None:
    """Lê em tons de cinza; funciona com acentos no caminho (Windows)."""
    try:
        dados = np.fromfile(str(caminho), dtype=np.uint8)
    except OSError:
        return None
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE)


def medir(img: np.ndarray) -> dict:
    """Tamanho, brilho, contraste, proporção de tinta e componentes conexos.

    A tinta é separada do fundo pelo limiar de Otsu. Se o fundo for escuro
    (tinta clara), a imagem é invertida antes, para medir sempre a tinta.
    """
    altura, largura = img.shape
    fundo_escuro = bool(np.median(img) < 128)
    tinta_escura = 255 - img if fundo_escuro else img
    _, binaria = cv2.threshold(tinta_escura, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    componentes = int((stats[1:, cv2.CC_STAT_AREA] >= 3).sum()) if n > 1 else 0
    return {
        "largura": largura,
        "altura": altura,
        "aspecto": largura / altura,
        "intensidade_media": float(img.mean()),
        "contraste": float(img.std()),
        "tinta": float((binaria > 0).mean()),
        "componentes": componentes,
        "fundo_escuro": fundo_escuro,
    }


def amostrar(df: pd.DataFrame, por_rotulo: int, semente: int = 0) -> pd.DataFrame:
    partes = [g.sample(min(len(g), por_rotulo), random_state=semente) for _, g in df.groupby("rotulo")]
    return pd.concat(partes) if partes else df.iloc[:0]


def medir_amostra(df: pd.DataFrame, p: Pastas, por_rotulo: int = 500) -> pd.DataFrame:
    from tqdm import tqdm

    amostra = amostrar(df, por_rotulo)
    linhas = []
    for _, linha in tqdm(amostra.iterrows(), total=len(amostra), desc="medindo imagens", unit="img"):
        img = ler_cinza(caminho_absoluto(linha["caminho"], p))
        if img is None or img.size == 0:
            continue
        linhas.append({"rotulo": linha["rotulo"], "split": linha["split"], "caminho": linha["caminho"], **medir(img)})
    return pd.DataFrame(linhas)


def repetidas(df: pd.DataFrame, p: Pastas) -> dict:
    """Arquivos com conteúdo idêntico (mesmo md5).

    Conta grupos repetidos, grupos que aparecem em mais de uma partição
    (vazamento entre treino e teste) e grupos com rótulos diferentes.
    """
    from tqdm import tqdm

    grupos = defaultdict(list)
    faltando = 0
    for caminho, split, rotulo in tqdm(
        zip(df["caminho"], df["split"], df["rotulo"]), total=len(df), desc="procurando repetidas", unit="img"
    ):
        try:
            with open(caminho_absoluto(caminho, p), "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
        except OSError:
            faltando += 1
            continue
        grupos[h].append((split, rotulo))
    repetidos = [g for g in grupos.values() if len(g) > 1]
    return {
        "imagens": len(df) - faltando,
        "arquivos_faltando": faltando,
        "grupos_repetidos": len(repetidos),
        "imagens_em_grupos_repetidos": sum(len(g) for g in repetidos),
        "grupos_em_mais_de_um_split": sum(len({s for s, _ in g}) > 1 for g in repetidos),
        "grupos_com_rotulos_diferentes": sum(len({r for _, r in g}) > 1 for g in repetidos),
    }


def teste_de_atalho(medidas: pd.DataFrame, semente: int = 0) -> dict | None:
    """O quanto medidas que não olham a forma da escrita já separam os rótulos.

    Usa regressão logística e uma árvore de profundidade 3, com validação
    cruzada estratificada em 5 partes, e compara com o acaso (1 / nº de rótulos).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text

    contagem = medidas["rotulo"].value_counts()
    contagem = contagem[contagem >= 5]
    if len(contagem) < 2:
        return None
    dados = medidas[medidas["rotulo"].isin(contagem.index)]
    X = dados[MEDIDAS].astype(float).to_numpy()
    y = dados["rotulo"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
    logistica = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    arvore = DecisionTreeClassifier(max_depth=3, random_state=semente)
    return {
        "rotulos": list(contagem.index),
        "acaso": 1 / len(contagem),
        "logistica": float(cross_val_score(logistica, X, y, cv=cv, scoring="balanced_accuracy").mean()),
        "arvore": float(cross_val_score(arvore, X, y, cv=cv, scoring="balanced_accuracy").mean()),
        "regras": export_text(arvore.fit(X, y), feature_names=MEDIDAS, decimals=3),
    }
