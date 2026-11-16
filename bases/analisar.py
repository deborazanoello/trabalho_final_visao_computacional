"""Análise exploratória das bases baixadas: tabelas, gráficos e um relatório.

Lê os índices de dados/indices/ e grava em relatorios/:
    analise_bases.md      relatório com tabelas e comentários
    figuras/*.png         contagens, medidas das imagens e exemplos

Exemplos:
    python -m bases.analisar
    python -m bases.analisar --amostra 1000 --sem-repetidas
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import graficos
from .bressay import ACIMA_ABAIXO, RISCADO
from .comum import RAIZ, carregar_indice, pastas
from .estatisticas import MEDIDAS, NOMES, medir_amostra, repetidas, teste_de_atalho

ORDEM_SPLITS = ["treino", "validacao", "teste", "sem_split"]

INDICES = {
    "bressay_palavras": {"titulo": "BRESSAY (palavras)", "imagens": True},
    "bressay_linhas": {"titulo": "BRESSAY (linhas)", "imagens": False},
    "dislexia": {"titulo": "Dyslexia Handwriting (letras)", "imagens": True},
    "rasuras_hf": {"titulo": "Handwritten cross-outs (palavras em inglês)", "imagens": True},
}

LEITURA_ATALHO = {
    "bressay_palavras": (
        "Aqui, separar bem não é necessariamente um problema: o risco acrescenta tinta e "
        "componentes à palavra. Mas tamanho e aspecto também separarem as classes indica que "
        "o modelo pode decidir pelo comprimento da palavra, e não pela rasura."
    ),
    "dislexia": (
        "Estas medidas não olham a forma da letra. Se elas já separam normal, espelhada e "
        "corrigida bem acima do acaso, as classes provavelmente diferem na origem da imagem "
        "(digitalização, tamanho, fundo), e um classificador pode aprender esse atalho em vez "
        "da reversão. Vale testar espelhando letras normais e vendo se o modelo as chama de espelhadas."
    ),
    "rasuras_hf": (
        "Quantidade de tinta e componentes devem mesmo separar palavras limpas das rasuradas. "
        "Separação também pela intensidade média ou pelo fundo indicaria diferença de captura "
        "(ferramenta de escrita), e não da rasura."
    ),
}


# ---------------------------------------------------------------- formatação


def _int(v) -> str:
    return f"{int(v):,}".replace(",", ".")


def _num(v, casas: int = 2) -> str:
    return f"{float(v):.{casas}f}".replace(".", ",")


def _pct(v) -> str:
    return f"{float(v):.1%}".replace(".", ",")


def _tabela(df: pd.DataFrame, formatos: dict | None = None, indice: str | None = None) -> str:
    formatos = formatos or {}
    df = df.reset_index() if indice else df
    if indice:
        df = df.rename(columns={df.columns[0]: indice})
    cab = "| " + " | ".join(NOMES.get(str(c), str(c)) for c in df.columns) + " |"
    sep = "|" + "|".join("---" if i == 0 else "---:" for i in range(len(df.columns))) + "|"
    corpo = []
    for _, linha in df.iterrows():
        celulas = []
        for c in df.columns:
            v = linha[c]
            f = formatos.get(c)
            if f:
                celulas.append(f(v))
            elif isinstance(v, bool):
                celulas.append("sim" if v else "não")
            elif isinstance(v, int) or (hasattr(v, "dtype") and str(getattr(v, "dtype", "")).startswith("int")):
                celulas.append(_int(v))
            elif isinstance(v, float):
                celulas.append(_num(v))
            else:
                celulas.append(str(v))
        corpo.append("| " + " | ".join(celulas) + " |")
    return "\n".join([cab, sep, *corpo])


def _ordenar_splits(colunas) -> list[str]:
    return [s for s in ORDEM_SPLITS if s in colunas] + [s for s in colunas if s not in ORDEM_SPLITS]


# ---------------------------------------------------------------- seções


def _contagens(nome, df, titulo, pasta_fig, md):
    tabela = pd.crosstab(df["rotulo"], df["split"])
    tabela = tabela[_ordenar_splits(tabela.columns)]
    fig = graficos.contagem_por_split(tabela, f"{titulo}: imagens por rótulo", pasta_fig / f"{nome}_contagem.png")
    tabela["total"] = tabela.sum(axis=1)
    tabela["% do total"] = tabela["total"] / tabela["total"].sum()
    tabela = tabela.sort_values("total", ascending=False)
    md.append("### Quantidade por rótulo e partição\n")
    md.append(f"{_int(len(df))} imagens no total.\n")
    md.append(_tabela(tabela, {"% do total": _pct}, indice="rótulo") + "\n")
    md.append(f"![contagem](figuras/{fig.name})\n")


def _marcas_bressay(df, md):
    marcas = ["riscada", "riscada_ilegivel", "correcao", "ilegivel"]
    linhas = []
    for split in _ordenar_splits(df["split"].unique()):
        parte = df[df["split"] == split]
        linha = {"partição": split, "imagens": len(parte)}
        for m in marcas:
            linha[m] = f"{_int(parte[m].sum())} ({_pct(parte[m].mean())})"
        linhas.append(linha)
    md.append("### Marcas na transcrição\n")
    md.append(
        "As marcas não são exclusivas: uma inserção riscada (`##--texto--##`) conta como "
        "riscada e como correção.\n"
    )
    md.append(_tabela(pd.DataFrame(linhas)) + "\n")


def _paginas_bressay(nome, df, pasta_fig, md):
    por_pagina = df.groupby("pagina").agg(palavras=("riscada", "size"), riscadas=("riscada", "sum"))
    por_pagina["proporcao"] = por_pagina["riscadas"] / por_pagina["palavras"]
    fig = graficos.histograma(
        por_pagina["proporcao"] * 100,
        "Palavras riscadas por página (cada página é um escritor)",
        "% de palavras riscadas na página",
        pasta_fig / f"{nome}_rasuras_por_pagina.png",
    )
    md.append("### Rasuras por página\n")
    md.append(
        "Cada página foi escrita por uma pessoa diferente, então esta distribuição mostra o "
        "quanto a rasura varia entre escritores.\n"
    )
    resumo = pd.DataFrame(
        [
            {
                "páginas": len(por_pagina),
                "sem nenhuma riscada": _pct((por_pagina["riscadas"] == 0).mean()),
                "mediana": _pct(por_pagina["proporcao"].median()),
                "percentil 90": _pct(por_pagina["proporcao"].quantile(0.9)),
                "máximo": _pct(por_pagina["proporcao"].max()),
            }
        ]
    )
    md.append(_tabela(resumo) + "\n")
    md.append(f"![rasuras por página](figuras/{fig.name})\n")


def _frequentes_bressay(df, md, n=20):
    riscadas, correcoes = Counter(), Counter()
    for texto in df["texto"].dropna().astype(str):
        for r in RISCADO.findall(texto):
            if r != "xxx" and "@@" not in r:
                riscadas[r.strip().lower()] += 1
        for acima, abaixo in ACIMA_ABAIXO.findall(texto):
            c = (acima or abaixo).strip()
            if "--" not in c and "@@" not in c:
                correcoes[c.lower()] += 1
    md.append(f"### Palavras riscadas e correções mais frequentes (legíveis, top {n})\n")
    linhas = []
    topo_r, topo_c = riscadas.most_common(n), correcoes.most_common(n)
    for i in range(max(len(topo_r), len(topo_c))):
        r = topo_r[i] if i < len(topo_r) else ("", "")
        c = topo_c[i] if i < len(topo_c) else ("", "")
        linhas.append(
            {
                "riscada": r[0],
                "vezes riscada": _int(r[1]) if r[1] != "" else "",
                "correção": c[0],
                "vezes corrigida": _int(c[1]) if c[1] != "" else "",
            }
        )
    if linhas:
        md.append(_tabela(pd.DataFrame(linhas)) + "\n")
    else:
        md.append("Nenhuma palavra riscada legível encontrada.\n")


def _ferramentas_hf(df, md):
    tabela = pd.crosstab(df["rotulo"], df["ferramenta"].fillna("(vazio)"))
    md.append("### Rótulo por ferramenta de escrita\n")
    md.append(_tabela(tabela, indice="rótulo") + "\n")


def _repetidas(df, p, md):
    r = repetidas(df, p)
    md.append("### Imagens repetidas\n")
    md.append(
        "Arquivos com conteúdo idêntico (mesmo md5). Repetidas em mais de uma partição "
        "vazam do treino para o teste; repetidas com rótulos diferentes são ruído no rótulo.\n"
    )
    md.append(_tabela(pd.DataFrame([{k: _int(v) for k, v in r.items()}])) + "\n")


def _medidas(nome, df, p, titulo, args, pasta_fig, md):
    medidas = medir_amostra(df, p, args.amostra)
    if medidas.empty:
        md.append("Nenhuma imagem pôde ser lida.\n")
        return
    medidas.to_csv(pasta_fig.parent / f"medidas_{nome}.csv", index=False)
    md.append(f"### Medidas das imagens (amostra de até {_int(args.amostra)} por rótulo)\n")
    md.append(
        "Tinta é a proporção de pixels de tinta depois do limiar de Otsu; componentes são as "
        "manchas de tinta conectadas; fundo escuro indica tinta clara sobre fundo escuro. "
        "A tabela mostra a mediana de cada rótulo.\n"
    )
    medianas = medidas.groupby("rotulo")[[m for m in MEDIDAS if m != "fundo_escuro"]].median()
    medianas["fundo_escuro"] = medidas.groupby("rotulo")["fundo_escuro"].mean()
    medianas.insert(0, "n", medidas.groupby("rotulo").size())
    formatos = {
        "n": _int,
        "largura": lambda v: _num(v, 0),
        "altura": lambda v: _num(v, 0),
        "aspecto": _num,
        "intensidade_media": lambda v: _num(v, 1),
        "contraste": lambda v: _num(v, 1),
        "tinta": _pct,
        "componentes": lambda v: _num(v, 0),
        "fundo_escuro": _pct,
    }
    md.append(_tabela(medianas, formatos, indice="rótulo") + "\n")
    fig = graficos.caixas_por_rotulo(
        medidas, [m for m in MEDIDAS if m != "fundo_escuro"], f"{titulo}: medidas por rótulo", pasta_fig / f"{nome}_medidas.png"
    )
    md.append(f"![medidas](figuras/{fig.name})\n")

    atalho = teste_de_atalho(medidas)
    md.append("### Teste de atalho: medidas simples já separam os rótulos?\n")
    if atalho is None:
        md.append("Poucos exemplos por rótulo para a validação cruzada.\n")
        return
    md.append(
        "Acurácia balanceada em validação cruzada (5 partes), usando só as medidas acima: "
        f"regressão logística **{_pct(atalho['logistica'])}**, árvore de profundidade 3 "
        f"**{_pct(atalho['arvore'])}**. Acaso: {_pct(atalho['acaso'])}.\n"
    )
    if nome in LEITURA_ATALHO:
        md.append(LEITURA_ATALHO[nome] + "\n")
    md.append("Regras da árvore:\n\n```\n" + atalho["regras"].rstrip() + "\n```\n")


# ---------------------------------------------------------------- principal


def analisar_indice(nome, df, p, args, pasta_fig) -> list[str]:
    cfg = INDICES[nome]
    titulo = cfg["titulo"]
    md = [f"## {titulo}\n"]
    _contagens(nome, df, titulo, pasta_fig, md)
    if nome.startswith("bressay"):
        _marcas_bressay(df, md)
    if nome == "bressay_palavras":
        _paginas_bressay(nome, df, pasta_fig, md)
        _frequentes_bressay(df, md)
    if nome == "rasuras_hf" and "ferramenta" in df.columns:
        _ferramentas_hf(df, md)
    if cfg["imagens"]:
        if not args.sem_repetidas:
            _repetidas(df, p, md)
        _medidas(nome, df, p, titulo, args, pasta_fig, md)
        fig = graficos.grade_de_exemplos(
            df, p, f"{titulo}: exemplos sorteados", pasta_fig / f"{nome}_exemplos.png", por_rotulo=args.exemplos
        )
        md.append("### Exemplos\n")
        md.append(f"![exemplos](figuras/{fig.name})\n")
    return md


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dados", help="pasta de dados (padrão: ./dados ou a variável TF_DADOS)")
    parser.add_argument("--saida", default=str(RAIZ / "relatorios"), help="pasta do relatório (padrão: ./relatorios)")
    parser.add_argument("--amostra", type=int, default=500, help="imagens medidas por rótulo (padrão: 500)")
    parser.add_argument("--exemplos", type=int, default=8, help="exemplos por rótulo na grade (padrão: 8)")
    parser.add_argument("--sem-repetidas", action="store_true", help="pula a busca de imagens repetidas")
    parser.add_argument("--indices", nargs="+", choices=list(INDICES), help="analisa só estes índices")
    args = parser.parse_args(argv)

    p = pastas(args.dados)
    saida = Path(args.saida).expanduser().resolve()
    pasta_fig = saida / "figuras"
    pasta_fig.mkdir(parents=True, exist_ok=True)

    nomes = args.indices or list(INDICES)
    encontrados = {n: carregar_indice(p, n) for n in nomes}
    faltando = [n for n, df in encontrados.items() if df is None or df.empty]
    presentes = {n: df for n, df in encontrados.items() if df is not None and not df.empty}
    if not presentes:
        print(f"Nenhum índice encontrado em {p.indices}. Rode antes: python -m bases.baixar", file=sys.stderr)
        return 1

    md = [
        "# Análise exploratória das bases\n",
        f"Gerado em {datetime.now():%d/%m/%Y %H:%M} com `python -m bases.analisar`.\n",
        "Índices analisados: " + ", ".join(f"`{n}`" for n in presentes) + ".\n",
    ]
    if faltando:
        md.append("Não encontrados (rode `python -m bases.baixar`): " + ", ".join(f"`{n}`" for n in faltando) + ".\n")
    for nome, df in presentes.items():
        print(f"\nAnalisando {nome} ({len(df):,} imagens)...")
        md.extend(analisar_indice(nome, df, p, args, pasta_fig))

    relatorio = saida / "analise_bases.md"
    relatorio.write_text("\n".join(md), encoding="utf-8")
    print(f"\nRelatório: {relatorio}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
