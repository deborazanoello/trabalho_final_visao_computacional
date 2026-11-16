"""Handwritten cross-outs (Hugging Face, CC BY 4.0): palavras em inglês.

Rótulos originais (coluna "rotulo"):
    clean                         palavra limpa
    single-line, double-line,     7 tipos de rasura
    diagonal, cross, wave,
    zig-zag, scratch
    mix                           subconjunto misto, com palavras limpas e rasuradas

A coluna "rasurada" vale True para os 7 tipos, False para clean e fica vazia
para mix, que mistura os dois casos.

As imagens são gravadas em dados/brutos/rasuras_hf/imagens/<split>/<rotulo>/
para que as três bases sejam lidas do mesmo jeito.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from .comum import Pastas, caminho_relativo, normalizar_split

HF_ID = "wahlinski/handwritten_cross-outs"
URL_PAGINA = f"https://huggingface.co/datasets/{HF_ID}"


def _carregar_hf() -> dict:
    """Partições da base, com as imagens em bytes (sem decodificar)."""
    from datasets import Image, load_dataset

    ds = load_dataset(HF_ID)
    return {nome: parte.cast_column("image", Image(decode=False)) for nome, parte in ds.items()}


def _bytes_e_extensao(imagem: dict) -> tuple[bytes, str]:
    dados = imagem.get("bytes")
    if dados is None:
        with open(imagem["path"], "rb") as f:
            dados = f.read()
    if dados[:8] == b"\x89PNG\r\n\x1a\n":
        return dados, ".png"
    if dados[:3] == b"\xff\xd8\xff":
        return dados, ".jpg"
    from PIL import Image

    saida = io.BytesIO()
    Image.open(io.BytesIO(dados)).save(saida, format="PNG")
    return saida.getvalue(), ".png"


def _rasurada(rotulo: str):
    if rotulo == "clean":
        return False
    if rotulo == "mix":
        return None
    return True


def preparar_e_indexar(p: Pastas, forcar: bool = False) -> pd.DataFrame:
    from tqdm import tqdm

    destino = p.brutos / "rasuras_hf" / "imagens"
    print(f"Baixando {HF_ID} pelo Hugging Face...")
    linhas = []
    for nome, parte in _carregar_hf().items():
        split = normalizar_split(nome)
        for i, ex in enumerate(tqdm(parte, desc=f"cross-outs ({split})", unit="img")):
            rotulo = str(ex["label"])
            pasta = destino / split / re.sub(r"[^\w-]", "_", rotulo)
            dados, extensao = _bytes_e_extensao(ex["image"])
            arquivo = pasta / f"{split}_{i:06d}{extensao}"
            if forcar or not arquivo.exists():
                pasta.mkdir(parents=True, exist_ok=True)
                arquivo.write_bytes(dados)
            linhas.append(
                {
                    "base": "rasuras_hf",
                    "split": split,
                    "rotulo": rotulo,
                    "rasurada": _rasurada(rotulo),
                    "ferramenta": ex.get("writing_tool"),
                    "texto": ex.get("text"),
                    "caminho": caminho_relativo(arquivo, p),
                }
            )
    return pd.DataFrame(linhas)
