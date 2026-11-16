"""Pastas, download, extração de zip e formato dos índices, usados por todas as bases."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

# Nomes de partição usados pelas bases, levados a um nome único em português.
SPLITS = {
    "training": "treino",
    "train": "treino",
    "validation": "validacao",
    "valid": "validacao",
    "val": "validacao",
    "test": "teste",
    "testing": "teste",
}

EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Pastas:
    """Onde ficam os arquivos baixados (brutos) e os índices em CSV."""

    dados: Path

    @property
    def brutos(self) -> Path:
        return self.dados / "brutos"

    @property
    def indices(self) -> Path:
        return self.dados / "indices"


def pastas(dados: str | Path | None = None) -> Pastas:
    """Pasta de dados: argumento, variável TF_DADOS ou ./dados na raiz do repositório."""
    base = dados or os.environ.get("TF_DADOS") or RAIZ / "dados"
    return Pastas(Path(base).expanduser().resolve())


def normalizar_split(nome: str) -> str:
    nome = nome.strip().lower()
    return SPLITS.get(nome, nome)


def caminho_relativo(arquivo: Path, p: Pastas) -> str:
    """Caminho guardado no índice: relativo à pasta de dados, com barras normais."""
    return Path(arquivo).resolve().relative_to(p.dados).as_posix()


def caminho_absoluto(relativo: str, p: Pastas) -> Path:
    return p.dados / relativo


def salvar_indice(df: pd.DataFrame, p: Pastas, nome: str) -> Path:
    p.indices.mkdir(parents=True, exist_ok=True)
    destino = p.indices / f"{nome}.csv"
    df.to_csv(destino, index=False)
    return destino


def carregar_indice(p: Pastas, nome: str) -> pd.DataFrame | None:
    origem = p.indices / f"{nome}.csv"
    if not origem.exists():
        return None
    # keep_default_na=False: palavras como "NA" ou "None" escritas à mão
    # continuam sendo texto, e não valores ausentes.
    return pd.read_csv(origem, keep_default_na=False, na_values=[""])


def md5_arquivo(arquivo: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(arquivo, "rb") as f:
        for parte in iter(lambda: f.read(bloco), b""):
            h.update(parte)
    return h.hexdigest()


def baixar_arquivo(url: str, destino: Path, md5: str | None = None) -> Path:
    """Baixa um arquivo grande, continuando de onde parou se a conexão cair."""
    import requests
    from tqdm import tqdm

    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".parcial")
    ja_baixado = parcial.stat().st_size if parcial.exists() else 0
    cabecalho = {"Range": f"bytes={ja_baixado}-"} if ja_baixado else {}

    with requests.get(url, stream=True, timeout=60, headers=cabecalho) as r:
        if r.status_code == 416:  # o arquivo parcial já está completo
            pass
        else:
            r.raise_for_status()
            continuar = r.status_code == 206
            if not continuar:
                ja_baixado = 0
            total = int(r.headers.get("content-length", 0)) + ja_baixado
            modo = "ab" if continuar else "wb"
            with open(parcial, modo) as f, tqdm(
                total=total or None, initial=ja_baixado, unit="B", unit_scale=True, desc=destino.name
            ) as barra:
                for bloco in r.iter_content(1 << 20):
                    f.write(bloco)
                    barra.update(len(bloco))

    if md5:
        obtido = md5_arquivo(parcial)
        if obtido != md5:
            parcial.unlink()
            raise RuntimeError(
                f"O arquivo baixado de {url} está corrompido (md5 {obtido}, esperado {md5}). "
                "Rode de novo para baixar outra vez."
            )
    parcial.replace(destino)
    return destino


class SenhaNecessaria(RuntimeError):
    """O zip é protegido por senha e nenhuma senha foi informada."""


def extrair_zip(arquivo: Path, destino: Path, senha: str | None = None) -> bool:
    """Extrai um zip (com ou sem senha) uma única vez.

    Devolve True se extraiu agora e False se já tinha sido extraído antes.
    """
    import pyzipper

    marcador = destino / f".extraido-{arquivo.name}.ok"
    if marcador.exists():
        return False
    destino.mkdir(parents=True, exist_ok=True)
    with pyzipper.AESZipFile(arquivo) as z:
        protegido = any(info.flag_bits & 0x1 for info in z.infolist())
        if protegido:
            if not senha:
                raise SenhaNecessaria(
                    f"{arquivo.name} é protegido por senha. Informe a senha com "
                    "--senha-dislexia ou com a variável de ambiente DISLEXIA_SENHA."
                )
            z.setpassword(senha.encode("utf-8"))
        try:
            z.extractall(destino)
        except RuntimeError as erro:
            raise RuntimeError(f"Não consegui extrair {arquivo.name}: senha incorreta? ({erro})") from erro
    marcador.touch()
    return True


def listar_imagens(pasta: Path):
    for arquivo in sorted(pasta.rglob("*")):
        if arquivo.suffix.lower() in EXTENSOES_IMAGEM and "__MACOSX" not in arquivo.parts:
            yield arquivo
