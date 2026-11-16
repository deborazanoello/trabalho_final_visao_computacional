"""BRESSAY: redações manuscritas em português do Brasil (Zenodo, CC BY 4.0).

O zip traz a pasta bressay/ com:
    data/{pages,paragraphs,lines,words}/<pagina>/**/*.png   imagem
    data/{...}/<pagina>/**/*.txt                             transcrição da imagem
    sets/{training,validation,test}.txt                      páginas de cada partição

A partição é feita por página: todas as palavras e linhas de uma página ficam
na mesma partição, então o mesmo escritor nunca aparece no treino e no teste.

Rótulos tirados das marcas da transcrição:
    --texto--   riscado e legível        --xxx--    riscado e ilegível
    ##texto##   escrito acima da linha   $$texto$$  escrito abaixo da linha
    @@???@@     ilegível
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .comum import Pastas, baixar_arquivo, caminho_relativo, extrair_zip

URL_PAGINA = "https://zenodo.org/records/11637681"
URL_ZIP = "https://zenodo.org/records/11637681/files/bressay.zip?download=1"
MD5_ZIP = "e4a4093304efff8322b605c8b79aa957"

# Nome em português usado nos índices -> nome da pasta na base.
NIVEIS = {
    "palavras": "words",
    "linhas": "lines",
    "paragrafos": "paragraphs",
    "paginas": "pages",
}
ARQUIVOS_SPLIT = {"treino": "training.txt", "validacao": "validation.txt", "teste": "test.txt"}

RISCADO = re.compile(r"--(.+?)--")
ACIMA_ABAIXO = re.compile(r"##(.+?)##|\$\$(.+?)\$\$")
ILEGIVEL = re.compile(r"@@\?\?\?@@")


def classificar_texto(texto: str) -> dict:
    """Marcas presentes na transcrição e um rótulo principal.

    As marcas não são exclusivas (``##--abc--##`` é uma inserção riscada).
    O rótulo principal segue a prioridade riscada > correcao > ilegivel > limpa.
    """
    riscados = RISCADO.findall(texto)
    marcas = {
        "riscada": bool(riscados),
        "riscada_ilegivel": any(r == "xxx" for r in riscados),
        "correcao": bool(ACIMA_ABAIXO.search(texto)),
        "ilegivel": bool(ILEGIVEL.search(texto)),
    }
    if marcas["riscada"]:
        rotulo = "riscada"
    elif marcas["correcao"]:
        rotulo = "correcao"
    elif marcas["ilegivel"]:
        rotulo = "ilegivel"
    else:
        rotulo = "limpa"
    return {**marcas, "rotulo": rotulo}


def localizar_raiz(pasta: Path) -> Path | None:
    """Pasta que contém sets/training.txt (normalmente .../bressay)."""
    if not pasta.exists():
        return None
    candidatas = [pasta, *pasta.glob("*"), *pasta.glob("*/*")]
    for c in candidatas:
        if (c / "sets" / "training.txt").is_file():
            return c
    return None


def preparar(p: Pastas, forcar: bool = False) -> Path:
    """Baixa (1,5 GB) e extrai a base, se ainda não estiver extraída."""
    destino = p.brutos / "bressay"
    raiz = localizar_raiz(destino)
    if raiz is not None and not forcar:
        return raiz
    arquivo = destino / "bressay.zip"
    if forcar or not arquivo.exists():
        print(f"Baixando o BRESSAY de {URL_PAGINA} (1,5 GB)...")
        baixar_arquivo(URL_ZIP, arquivo, md5=MD5_ZIP)
    if forcar:
        for marcador in destino.glob(".extraido-*"):
            marcador.unlink()
    print("Extraindo bressay.zip...")
    extrair_zip(arquivo, destino)
    raiz = localizar_raiz(destino)
    if raiz is None:
        raise RuntimeError(f"Extraí {arquivo}, mas não encontrei sets/training.txt dentro dele.")
    return raiz


def indexar(raiz: Path, p: Pastas, nivel: str = "palavras") -> pd.DataFrame:
    """Uma linha por imagem do nível pedido, com a transcrição e as marcas."""
    from tqdm import tqdm

    pasta_nivel = raiz / "data" / NIVEIS[nivel]
    linhas, paginas_faltando = [], []
    for split, nome_arquivo in ARQUIVOS_SPLIT.items():
        paginas = [x.strip() for x in (raiz / "sets" / nome_arquivo).read_text().splitlines() if x.strip()]
        for pagina in tqdm(paginas, desc=f"BRESSAY {nivel} ({split})", unit="página"):
            pasta = pasta_nivel / pagina
            if not pasta.is_dir():
                paginas_faltando.append(pagina)
                continue
            for imagem in sorted(pasta.rglob("*.png")):
                transcricao = imagem.with_suffix(".txt")
                texto = (
                    transcricao.read_text(encoding="utf-8", errors="replace").strip()
                    if transcricao.exists()
                    else ""
                )
                linhas.append(
                    {
                        "base": "bressay",
                        "nivel": nivel,
                        "split": split,
                        "pagina": pagina,
                        "caminho": caminho_relativo(imagem, p),
                        "texto": texto,
                        **classificar_texto(texto),
                    }
                )
    if paginas_faltando:
        print(f"Aviso: {len(paginas_faltando)} páginas listadas em sets/ não têm pasta em {pasta_nivel}.")
    return pd.DataFrame(linhas)
