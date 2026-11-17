# %% [markdown]
# # 1 - Preparar as fotos
# Lê as fotos de `dados/fotos/<celular>/<livro>/`, corrige a rotação do celular,
# passa para cinza, reduz para `LADO_MAX` e grava em `saida/fotos_prontas/`.
# Também monta `saida/indice_fotos.csv`, com celular, livro, página e condição de cada foto.
#
# Nome das fotos: `p012_ref.heic` (referência: página plana, de frente, boa luz),
# `p012_inclinada.heic`, `p012_sombra.heic`, `p012_curva.heic` (uso real).

# %% ===== 1 - Imports =====
import time

import cv2
import matplotlib.pyplot as plt
import pandas as pd

import comum
from comum import INDICE_FOTOS, PASTA_FOTOS, PASTA_PRONTAS

# %% ===== 2 - Encontrar as fotos =====
if not PASTA_FOTOS.exists():
    raise FileNotFoundError(f"Não achei a pasta: {PASTA_FOTOS}")

arquivos = sorted(p for p in PASTA_FOTOS.rglob("*") if p.is_file() and p.suffix.lower() in comum.EXTENSOES)
fora_do_padrao = [p for p in arquivos if comum.info_do_nome(p) is None]

print(len(arquivos), "fotos encontradas em", PASTA_FOTOS)
if fora_do_padrao:
    print(len(fora_do_padrao), "fora do padrão fotos/<celular>/<livro>/p012_condicao (serão ignoradas):")
    for p in fora_do_padrao[:10]:
        print("   ", p.relative_to(PASTA_FOTOS))


# %% ===== 3 - Preparar e gravar =====
def nitidez(cinza):
    """Variância do laplaciano: quanto menor, mais borrada a foto."""
    return float(cv2.Laplacian(cinza, cv2.CV_64F).var())


linhas = []
inicio = time.time()
for i, p in enumerate(arquivos, start=1):
    info = comum.info_do_nome(p)
    if info is None:
        continue
    destino = PASTA_PRONTAS / info["celular"] / info["livro"] / (p.stem + ".png")
    try:
        original = comum.ler_foto_cinza(p)
        pronta = comum.reduzir(original)
        comum.salvar_png(destino, pronta)
    except Exception as erro:
        print("problema em", p.name, "->", erro)
        continue
    linhas.append({
        **info,
        "arquivo_original": p.relative_to(comum.DADOS).as_posix(),
        "arquivo_pronto": destino.relative_to(comum.SAIDA).as_posix(),
        "largura_original": original.shape[1],
        "altura_original": original.shape[0],
        "largura": pronta.shape[1],
        "altura": pronta.shape[0],
        "brilho": float(pronta.mean()),
        "contraste": float(pronta.std()),
        "nitidez": nitidez(pronta),
    })
    if i % 10 == 0:
        print("fotos:", i, "de", len(arquivos), f"({time.time() - inicio:.0f}s)")

indice = pd.DataFrame(linhas)
INDICE_FOTOS.parent.mkdir(parents=True, exist_ok=True)
indice.to_csv(INDICE_FOTOS, index=False)
print("\n", len(indice), "fotos prontas em", PASTA_PRONTAS)
print("Índice:", INDICE_FOTOS)

# %% ===== 4 - Conferir =====
print("Fotos por celular, livro e condição:")
print(pd.crosstab([indice["celular"], indice["livro"]], indice["condicao"]))

tem_ref = indice.assign(ref=indice["condicao"] == "ref").groupby(["celular", "livro", "pagina"])["ref"].any()
sem_ref = tem_ref[~tem_ref]
if len(sem_ref):
    print("\nPáginas sem foto de referência (p..._ref):")
    for celular, livro, pagina in sem_ref.index:
        print(f"    {celular} / {livro} / página {pagina}")
else:
    print("\nTodas as páginas têm foto de referência.")

# %% ===== 5 - Figura: uma página em todas as condições =====
exemplo = indice[indice["condicao"] == "ref"].iloc[0]
mesma_pagina = indice[(indice["livro"] == exemplo["livro"]) & (indice["pagina"] == exemplo["pagina"])]
mesma_pagina = mesma_pagina.sort_values(["celular", "condicao"], key=lambda s: s.where(s != "ref", ""))

fig, ax = plt.subplots(1, len(mesma_pagina), figsize=(3.2 * len(mesma_pagina), 4.6), squeeze=False)
for a, (_, linha) in zip(ax[0], mesma_pagina.iterrows()):
    a.imshow(comum.ler_png(comum.SAIDA / linha["arquivo_pronto"]), cmap="gray", vmin=0, vmax=255)
    a.set_title(f"{linha['celular']}\n{linha['condicao']}", fontsize=9)
    a.axis("off")
fig.suptitle(f"{exemplo['livro']}, página {exemplo['pagina']}")
plt.tight_layout()
comum.mostrar(fig, "exemplos_fotos.png")
