# %% [markdown]
# # 2 - Calibrar a câmera de cada celular
# Lê as fotos de um tabuleiro de xadrez em `dados/calibracao/<celular>/` e calcula a matriz
# da câmera e a distorção da lente. Grava `saida/calibracao/<celular>.npz`.
#
# Como fotografar: imprima um tabuleiro (ou abra na tela do computador), cole numa superfície
# plana e tire de 15 a 20 fotos de ângulos e distâncias diferentes, com o tabuleiro inteiro
# aparecendo. Use a MESMA orientação (retrato) das fotos das páginas.

# %% ===== 1 - Imports =====
import cv2
import matplotlib.pyplot as plt
import numpy as np

import comum
from comum import PASTA_CALIBRACAO

# %% ===== 2 - Configurações =====
TABULEIRO = (9, 6)    # cantos INTERNOS do tabuleiro (um tabuleiro de 10 x 7 quadrados)
QUADRADO_MM = 25.0    # lado de um quadrado; só muda a escala, não a matriz da câmera
MIN_FOTOS = 5         # mínimo de fotos com o tabuleiro encontrado

if not PASTA_CALIBRACAO.exists():
    raise FileNotFoundError(f"Não achei a pasta: {PASTA_CALIBRACAO}")

# coordenadas dos cantos no plano do tabuleiro (z = 0)
objp = np.zeros((TABULEIRO[0] * TABULEIRO[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:TABULEIRO[0], 0:TABULEIRO[1]].T.reshape(-1, 2) * QUADRADO_MM
criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# %% ===== 3 - Encontrar os cantos e calibrar =====
resultados = {}
for pasta in sorted(p for p in PASTA_CALIBRACAO.iterdir() if p.is_dir()):
    celular = pasta.name
    fotos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in comum.EXTENSOES)
    pontos_3d, pontos_2d, tamanho, exemplo = [], [], None, None

    for p in fotos:
        img = comum.reduzir(comum.ler_foto_cinza(p))
        if tamanho is None:
            tamanho = (img.shape[1], img.shape[0])
        if (img.shape[1], img.shape[0]) != tamanho:
            print(f"  {p.name}: tamanho diferente das outras fotos (outra orientação?), ignorada")
            continue
        achou, cantos = cv2.findChessboardCorners(
            img, TABULEIRO, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not achou:
            print(f"  {p.name}: tabuleiro não encontrado")
            continue
        cantos = cv2.cornerSubPix(img, cantos, (11, 11), (-1, -1), criterio)
        pontos_3d.append(objp)
        pontos_2d.append(cantos)
        if exemplo is None:
            exemplo = (img, cantos)

    if len(pontos_2d) < MIN_FOTOS:
        print(f"{celular}: só {len(pontos_2d)} de {len(fotos)} fotos com o tabuleiro; preciso de {MIN_FOTOS}")
        continue

    erro, K, dist, _, _ = cv2.calibrateCamera(pontos_3d, pontos_2d, tamanho, None, None)
    destino = comum.arquivo_calibracao(celular)
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destino, K=K, dist=dist, tamanho=np.array(tamanho), erro=erro, n_fotos=len(pontos_2d))
    resultados[celular] = (K, dist, tamanho, exemplo)
    print(f"{celular}: {len(pontos_2d)} de {len(fotos)} fotos usadas, erro de reprojeção = {erro:.3f} px")
    print(f"    fx = {K[0, 0]:.1f}  fy = {K[1, 1]:.1f}  cx = {K[0, 2]:.1f}  cy = {K[1, 2]:.1f}")
    print(f"    distorção (k1, k2, p1, p2, k3) = {np.round(dist.ravel(), 4)}")

# %% ===== 4 - Figura: cantos encontrados e foto corrigida =====
for celular, (K, dist, tamanho, (img, cantos)) in resultados.items():
    com_cantos = cv2.drawChessboardCorners(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), TABULEIRO, cantos, True)
    corrigida = comum.corrigir_distorcao(img, (K, dist, tamanho))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(com_cantos)
    ax[0].set_title("Cantos encontrados")
    ax[1].imshow(corrigida, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("Distorção corrigida")
    for a in ax:
        a.axis("off")
    fig.suptitle(celular)
    plt.tight_layout()
    comum.mostrar(fig, f"calibracao_{celular}.png")
