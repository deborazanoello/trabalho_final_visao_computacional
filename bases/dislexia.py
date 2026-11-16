"""Dyslexia Handwriting Dataset (Kaggle): letras isoladas em três classes.

    Normal     letra na orientação padrão          -> rótulo "normal"
    Reversal   letra espelhada                     -> rótulo "espelhada"
    Corrected  letra que começou espelhada e foi   -> rótulo "corrigida"
               corrigida

As imagens vêm em Gambo.zip, organizadas em Train/ e Test/, uma pasta por
classe. O Gambo.zip é protegido por senha: procure a senha na página da base
no Kaggle e informe com --senha-dislexia ou com a variável DISLEXIA_SENHA.

As letras vêm de fontes diferentes (NIST SD19, outra base do Kaggle e crianças
em Penang, na Malásia). A análise verifica se estatísticas simples, como
tamanho e cor de fundo, já separam as classes, o que indicaria um atalho.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .comum import Pastas, SPLITS, caminho_relativo, extrair_zip, listar_imagens, normalizar_split

KAGGLE_ID = "drizasazanitaisa/dyslexia-handwriting-dataset"
URL_PAGINA = f"https://www.kaggle.com/datasets/{KAGGLE_ID}"

CLASSES = {
    "normal": "normal",
    "reversal": "espelhada",
    "reversed": "espelhada",
    "corrected": "corrigida",
}


def _baixar_kaggle() -> Path:
    try:
        import kagglehub

        return Path(kagglehub.dataset_download(KAGGLE_ID))
    except Exception as erro:  # sem login, sem rede, etc.
        raise RuntimeError(
            f"Não consegui baixar pelo kagglehub ({erro}).\n"
            f"Baixe pelo site ({URL_PAGINA}, botão Download) e rode de novo com\n"
            "  python -m bases.baixar dislexia --arquivo-dislexia caminho/do/archive.zip"
        ) from erro


def _extrair_zips(pasta: Path, senha: str | None, rodadas: int = 3) -> None:
    """Extrai os zips dentro da pasta, inclusive zips dentro de zips."""
    for _ in range(rodadas):
        extraiu = False
        for arquivo in sorted(pasta.rglob("*.zip")):
            if "__MACOSX" in arquivo.parts:
                continue
            extraiu |= extrair_zip(arquivo, arquivo.parent / arquivo.stem, senha)
        if not extraiu:
            return


def preparar(p: Pastas, senha: str | None = None, arquivo: str | Path | None = None, forcar: bool = False) -> Path:
    """Baixa (ou usa o arquivo informado) e extrai tudo em dados/brutos/dislexia."""
    destino = p.brutos / "dislexia"
    if forcar and destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True, exist_ok=True)

    origem = Path(arquivo).expanduser().resolve() if arquivo else None
    ja_preparado = any(destino.rglob(".extraido-*")) or any(listar_imagens(destino))
    if origem is None and not ja_preparado:
        print(f"Baixando {KAGGLE_ID} pelo kagglehub...")
        origem = _baixar_kaggle()

    if origem is not None:
        if origem.is_file():
            shutil.copy2(origem, destino / origem.name)
        else:
            shutil.copytree(origem, destino / "kaggle", dirs_exist_ok=True)

    _extrair_zips(destino, senha)
    if not any(listar_imagens(destino)):
        raise RuntimeError(f"Nenhuma imagem encontrada em {destino} depois da extração.")
    return destino


def indexar(pasta: Path, p: Pastas) -> pd.DataFrame:
    """Uma linha por letra; classe e partição saem dos nomes das pastas."""
    linhas, sem_classe = [], 0
    for imagem in listar_imagens(pasta):
        partes = [x.lower() for x in imagem.relative_to(pasta).parts[:-1]]
        classe = next((x for x in reversed(partes) if x in CLASSES), None)
        if classe is None:
            sem_classe += 1
            continue
        split = next((normalizar_split(x) for x in reversed(partes) if x in SPLITS), "sem_split")
        linhas.append(
            {
                "base": "dislexia",
                "split": split,
                "classe_original": classe,
                "rotulo": CLASSES[classe],
                "caminho": caminho_relativo(imagem, p),
            }
        )
    if sem_classe:
        print(f"Aviso: {sem_classe} imagens fora das pastas Normal/Reversal/Corrected foram ignoradas.")
    return pd.DataFrame(linhas)
