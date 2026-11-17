# %% [markdown]
# # 3 - Reconhecer a página e endireitar a foto
# Para cada foto de uso real, compara com as fotos de referência do mesmo livro usando
# características locais (SIFT ou ORB) e RANSAC. A referência com mais pontos em comum
# diz qual é a página; a homografia encontrada endireita a foto no enquadramento da referência.
#
# Grava `saida/retificadas/` e `saida/reconhecimento.csv` (página verdadeira x reconhecida).

# %% ===== 1 - Imports =====
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import comum
from comum import INDICE_FOTOS, SAIDA

# %% ===== 2 - Configurações =====
METODO = "SIFT"          # "SIFT" ou "ORB"
N_PONTOS = 4000          # máximo de pontos por imagem
RAZAO_LOWE = 0.75        # teste da razão: descarta correspondências ambíguas
LIMIAR_RANSAC = 5.0      # distância máxima (px) para um ponto contar como inlier
MIN_INLIERS = 30         # abaixo disso, a página é dada como "não reconhecida"
CORRIGIR_DISTORCAO = True

PASTA_RETIFICADAS = SAIDA / "retificadas"
RESULTADO = SAIDA / "reconhecimento.csv"

indice = pd.read_csv(INDICE_FOTOS)
referencias = indice[indice["condicao"] == "ref"]
consultas = indice[indice["condicao"] != "ref"]
print(len(referencias), "referências e", len(consultas), "fotos para reconhecer")

# %% ===== 3 - Funções =====
if METODO == "SIFT":
    detector = cv2.SIFT_create(N_PONTOS)
    casador = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})  # busca aproximada, rápida
else:
    detector = cv2.ORB_create(N_PONTOS)
    casador = cv2.BFMatcher(cv2.NORM_HAMMING)


def carregar(linha):
    img = comum.ler_png(SAIDA / linha["arquivo_pronto"])
    if CORRIGIR_DISTORCAO:
        calibracao = comum.carregar_calibracao(linha["celular"])
        if calibracao is not None:
            img = comum.corrigir_distorcao(img, calibracao)
    return img


def casar(des_foto, des_ref):
    """Correspondências que passam no teste da razão de Lowe."""
    if des_foto is None or des_ref is None or len(des_ref) < 2:
        return []
    pares = casador.knnMatch(des_foto, des_ref, k=2)
    return [m for m, n in (p for p in pares if len(p) == 2) if m.distance < RAZAO_LOWE * n.distance]


def homografia(kp_foto, kp_ref, bons):
    """Homografia foto -> referência por RANSAC, com o número de inliers."""
    if len(bons) < 4:
        return None, None, 0
    origem = np.float32([kp_foto[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
    destino = np.float32([kp_ref[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)
    H, mascara = cv2.findHomography(origem, destino, cv2.RANSAC, LIMIAR_RANSAC)
    if H is None:
        return None, None, 0
    return H, mascara.ravel().astype(bool), int(mascara.sum())


# %% ===== 4 - Reconhecer cada foto =====
linhas, exemplo = [], None
for livro, consultas_livro in consultas.groupby("livro"):
    refs_livro = referencias[referencias["livro"] == livro]
    if refs_livro.empty:
        print(f"{livro}: nenhuma referência, fotos ignoradas")
        continue

    # pontos das referências, calculados uma vez por livro
    base = []
    for _, ref in refs_livro.iterrows():
        img_ref = carregar(ref)
        kp, des = detector.detectAndCompute(img_ref, None)
        base.append((ref, img_ref, kp, des))
    print(f"{livro}: {len(base)} referências descritas")

    for _, foto in consultas_livro.iterrows():
        img = carregar(foto)
        kp, des = detector.detectAndCompute(img, None)
        tentativas = []
        for ref, img_ref, kp_ref, des_ref in base:
            bons = casar(des, des_ref)
            H, mascara, n = homografia(kp, kp_ref, bons)
            tentativas.append((n, ref, img_ref, kp_ref, bons, H, mascara))
        tentativas.sort(key=lambda t: t[0], reverse=True)
        n, ref, img_ref, kp_ref, bons, H, mascara = tentativas[0]
        segundo = tentativas[1][0] if len(tentativas) > 1 else 0
        reconhecida = int(ref["pagina"]) if n >= MIN_INLIERS else None

        if reconhecida is not None:
            retificada = cv2.warpPerspective(img, H, (img_ref.shape[1], img_ref.shape[0]))
            nome = foto["arquivo_pronto"].split("/")[-1]
            comum.salvar_png(PASTA_RETIFICADAS / foto["celular"] / livro / nome, retificada)
            if exemplo is None:
                exemplo = (foto, img, kp, ref, img_ref, kp_ref, bons, mascara, retificada)

        linhas.append({
            "arquivo": foto["arquivo_pronto"],
            "celular": foto["celular"],
            "livro": livro,
            "condicao": foto["condicao"],
            "pagina": int(foto["pagina"]),
            "pagina_reconhecida": reconhecida,
            "inliers": n,
            "inliers_segunda": segundo,
            "pontos_foto": len(kp),
            "acertou": reconhecida == int(foto["pagina"]),
        })

resultado = pd.DataFrame(linhas)
resultado.to_csv(RESULTADO, index=False)
print("\nResultado:", RESULTADO)

# %% ===== 5 - Resumo =====
print(f"Acerto geral: {100 * resultado['acertou'].mean():.1f}% de {len(resultado)} fotos\n")
print("Acerto (%) por condição e celular:")
print((100 * resultado.pivot_table(index="condicao", columns="celular", values="acertou", aggfunc="mean")).round(1))
erradas = resultado[~resultado["acertou"]]
if len(erradas):
    print("\nFotos não reconhecidas ou trocadas:")
    print(erradas[["arquivo", "pagina", "pagina_reconhecida", "inliers", "inliers_segunda"]].to_string(index=False))

# %% ===== 6 - Figura: um exemplo =====
if exemplo is not None:
    foto, img, kp, ref, img_ref, kp_ref, bons, mascara, retificada = exemplo
    inliers = [m for m, ok in zip(bons, mascara) if ok][:80]
    pares = cv2.drawMatches(img, kp, img_ref, kp_ref, inliers, None,
                            matchColor=(32, 80, 104), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, ax = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1, 1]})
    ax[0].imshow(pares)
    ax[0].set_title(f"Correspondências ({len(inliers)} de {int(mascara.sum())} inliers)")
    ax[1].imshow(retificada, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title(f"Foto endireitada ({foto['condicao']})")
    ax[2].imshow(img_ref, cmap="gray", vmin=0, vmax=255)
    ax[2].set_title(f"Referência: página {ref['pagina']}")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    comum.mostrar(fig, "exemplo_reconhecimento.png")
