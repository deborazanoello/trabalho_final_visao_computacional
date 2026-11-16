"""Gráficos da análise exploratória (PNG, matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .comum import Pastas, caminho_absoluto  # noqa: E402
from .estatisticas import NOMES, ler_cinza  # noqa: E402

# Ordem fixa de cores por partição (paleta categórica validada para daltonismo).
CORES_SPLIT = {"treino": "#2a78d6", "validacao": "#eb6834", "teste": "#1baf7a", "sem_split": "#eda100"}
COR_UNICA = "#2a78d6"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRADE = "#e6e5e1"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": GRADE,
        "axes.labelcolor": TEXTO_SECUNDARIO,
        "axes.titlecolor": TEXTO,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXTO_SECUNDARIO,
        "ytick.color": TEXTO_SECUNDARIO,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def _salvar(fig, destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destino


def contagem_por_split(tabela: pd.DataFrame, titulo: str, destino: Path) -> Path:
    """Barras horizontais: um grupo por rótulo, uma barra por partição."""
    splits = [s for s in CORES_SPLIT if s in tabela.columns] + [s for s in tabela.columns if s not in CORES_SPLIT]
    rotulos = list(tabela.index)
    altura_barra = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rotulos) * max(len(splits), 1) + 1.2))
    y = np.arange(len(rotulos))
    for i, split in enumerate(splits):
        ax.barh(
            y + i * altura_barra,
            tabela[split].to_numpy(),
            height=altura_barra * 0.9,
            color=CORES_SPLIT.get(split, "#6250d6"),
            label=split,
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_yticks(y + altura_barra * (len(splits) - 1) / 2, rotulos)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("imagens")
    ax.set_title(titulo, loc="left")
    if len(splits) > 1:
        ax.legend(loc="lower right")
    return _salvar(fig, destino)


def _caixas_horizontais(ax, valores, rotulos, **opcoes):
    try:  # matplotlib >= 3.10
        return ax.boxplot(valores, orientation="horizontal", tick_labels=rotulos, **opcoes)
    except TypeError:  # versões anteriores
        return ax.boxplot(valores, vert=False, labels=rotulos, **opcoes)


def caixas_por_rotulo(medidas: pd.DataFrame, colunas: list[str], titulo: str, destino: Path) -> Path:
    """Uma caixa por rótulo para cada medida (pequenos múltiplos)."""
    rotulos = sorted(medidas["rotulo"].unique())
    ncol = 4
    nlin = int(np.ceil(len(colunas) / ncol))
    fig, eixos = plt.subplots(nlin, ncol, figsize=(12, 2.6 * nlin + 0.4 * len(rotulos) / 3), squeeze=False)
    for ax, coluna in zip(eixos.flat, colunas):
        valores = [medidas.loc[medidas["rotulo"] == r, coluna].astype(float).to_numpy() for r in rotulos]
        _caixas_horizontais(
            ax,
            valores,
            rotulos,
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#cfe0f5", "edgecolor": COR_UNICA, "linewidth": 1},
            medianprops={"color": COR_UNICA, "linewidth": 2},
            whiskerprops={"color": COR_UNICA, "linewidth": 1},
            capprops={"color": COR_UNICA, "linewidth": 1},
        )
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(NOMES.get(coluna, coluna), loc="left")
    for ax in list(eixos.flat)[len(colunas):]:
        ax.set_visible(False)
    fig.suptitle(titulo, x=0.01, ha="left", color=TEXTO, fontsize=11)
    fig.tight_layout()
    return _salvar(fig, destino)


def grade_de_exemplos(df: pd.DataFrame, p: Pastas, titulo: str, destino: Path, por_rotulo: int = 8, semente: int = 0) -> Path:
    """Exemplos sorteados: uma linha por rótulo."""
    rotulos = sorted(df["rotulo"].unique())
    fig, eixos = plt.subplots(
        len(rotulos), por_rotulo, figsize=(1.5 * por_rotulo, 1.1 * len(rotulos) + 0.5), squeeze=False
    )
    for i, rotulo in enumerate(rotulos):
        grupo = df[df["rotulo"] == rotulo]
        escolhidos = grupo.sample(min(len(grupo), por_rotulo), random_state=semente)
        for j, ax in enumerate(eixos[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for lado in ax.spines.values():
                lado.set_visible(False)
            if j < len(escolhidos):
                img = ler_cinza(caminho_absoluto(escolhidos.iloc[j]["caminho"], p))
                if img is not None:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if j == 0:
                ax.set_ylabel(rotulo, rotation=0, ha="right", va="center", color=TEXTO, fontsize=9)
    fig.suptitle(titulo, x=0.01, ha="left", color=TEXTO, fontsize=11)
    fig.tight_layout()
    return _salvar(fig, destino)


def histograma(valores, titulo: str, rotulo_x: str, destino: Path, faixas: int = 30) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(valores, bins=faixas, color=COR_UNICA, edgecolor="white", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("páginas")
    ax.set_title(titulo, loc="left")
    return _salvar(fig, destino)
