"""Baixa as bases e grava um índice CSV para cada uma em dados/indices/.

Exemplos:
    python -m bases.baixar                       # as três bases
    python -m bases.baixar bressay rasuras_hf    # só algumas
    python -m bases.baixar dislexia --senha-dislexia SENHA
    python -m bases.baixar dislexia --arquivo-dislexia ~/Downloads/archive.zip
"""

from __future__ import annotations

import argparse
import os
import sys

from . import bressay, dislexia, rasuras_hf
from .comum import pastas, salvar_indice

BASES = ("bressay", "dislexia", "rasuras_hf")


def _bressay(p, args):
    raiz = bressay.preparar(p, forcar=args.forcar)
    resumo = []
    for nivel in args.niveis_bressay:
        df = bressay.indexar(raiz, p, nivel)
        destino = salvar_indice(df, p, f"bressay_{nivel}")
        resumo.append(f"{len(df):,} {nivel} -> {destino}")
    return "; ".join(resumo)


def _dislexia(p, args):
    pasta = dislexia.preparar(p, senha=args.senha_dislexia, arquivo=args.arquivo_dislexia, forcar=args.forcar)
    df = dislexia.indexar(pasta, p)
    return f"{len(df):,} letras -> {salvar_indice(df, p, 'dislexia')}"


def _rasuras_hf(p, args):
    df = rasuras_hf.preparar_e_indexar(p, forcar=args.forcar)
    return f"{len(df):,} palavras -> {salvar_indice(df, p, 'rasuras_hf')}"


ACOES = {"bressay": _bressay, "dislexia": _dislexia, "rasuras_hf": _rasuras_hf}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bases", nargs="*", help=f"bases a baixar: {', '.join(BASES)} (padrão: todas)")
    parser.add_argument("--dados", help="pasta de dados (padrão: ./dados ou a variável TF_DADOS)")
    parser.add_argument(
        "--niveis-bressay",
        nargs="+",
        default=["palavras", "linhas"],
        choices=list(bressay.NIVEIS),
        help="níveis do BRESSAY a indexar (padrão: palavras linhas)",
    )
    parser.add_argument(
        "--senha-dislexia",
        default=os.environ.get("DISLEXIA_SENHA"),
        help="senha do Gambo.zip (ou variável DISLEXIA_SENHA)",
    )
    parser.add_argument("--arquivo-dislexia", help="zip ou pasta baixados manualmente do Kaggle")
    parser.add_argument("--forcar", action="store_true", help="baixa e extrai de novo, mesmo que já exista")
    args = parser.parse_args(argv)

    escolhidas = args.bases or list(BASES)
    invalidas = [b for b in escolhidas if b not in BASES]
    if invalidas:
        parser.error(f"base desconhecida: {', '.join(invalidas)}. Opções: {', '.join(BASES)}")

    p = pastas(args.dados)
    print(f"Pasta de dados: {p.dados}\n")
    falhas = 0
    for base in escolhidas:
        try:
            print(f"OK  {base}: {ACOES[base](p, args)}\n")
        except Exception as erro:
            falhas += 1
            print(f"ERRO {base}: {erro}\n", file=sys.stderr)
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
