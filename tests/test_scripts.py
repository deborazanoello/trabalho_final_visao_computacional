"""Roda os scripts com fotos sintéticas, sem precisar das fotos reais.

    python -m pytest
"""

import os
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw, ImageFont

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# Texto parecido com o de um livro: palavras comuns que se repetem em toda página
# e um vocabulário grande de palavras que variam de página para página.
COMUNS = "de a o que e do da em um para com nem uma os no se na por mais as dos como mas".split()
_rng = random.Random(0)
VARIADAS = ["".join(_rng.choice("abcdefghijlmnoprstuv") for _ in range(_rng.randint(3, 10)))
            for _ in range(3000)]


def _fonte(tamanho):
    try:
        return ImageFont.load_default(size=tamanho)
    except TypeError:  # Pillow antigo
        return ImageFont.load_default()


def pagina_sintetica(semente, largura=900, altura=1300):
    """Uma página de texto com palavras sorteadas (cada semente dá um texto diferente)."""
    rng = random.Random(semente)
    img = Image.new("L", (largura, altura), 245)
    desenho = ImageDraw.Draw(img)
    fonte = _fonte(30)
    for i in range(30):
        linha = " ".join(rng.choice(COMUNS) if rng.random() < 0.45 else rng.choice(VARIADAS)
                         for _ in range(rng.randint(5, 8)))
        desenho.text((70, 70 + i * 38), linha, fill=20, font=fonte)
    desenho.text((largura // 2 - 20, altura - 60), str(semente), fill=20, font=fonte)
    return np.array(img)


def foto_inclinada(pagina, semente):
    """A página vista de lado, sobre um fundo escuro."""
    rng = np.random.default_rng(semente)
    h, w = pagina.shape
    cantos = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    deslocados = cantos + rng.uniform(-70, 70, cantos.shape).astype(np.float32) + np.float32([150, 150])
    H = cv2.getPerspectiveTransform(cantos, deslocados)
    return cv2.warpPerspective(pagina, H, (w + 300, h + 300), borderValue=60)


def foto_com_sombra(pagina):
    h, w = pagina.shape
    gradiente = np.linspace(0.45, 1.0, w, dtype=np.float32)[None, :]
    return (pagina.astype(np.float32) * gradiente).astype(np.uint8)


def salvar_jpg(caminho, img, orientacao=None):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(img)
    if orientacao:
        exif = Image.Exif()
        exif[0x0112] = orientacao
        pil.save(caminho, quality=92, exif=exif)
    else:
        pil.save(caminho, quality=92)


def tabuleiro_visto_pela_camera(K, rx, ry, tz, tamanho=(1600, 1200)):
    """Tabuleiro de 10 x 7 quadrados (40 px cada) projetado por uma câmera pinhole."""
    q, margem = 40, 40
    tab = np.full((7 * q + 2 * margem, 10 * q + 2 * margem), 255, np.uint8)
    for i in range(7):
        for j in range(10):
            if (i + j) % 2 == 0:
                tab[margem + i * q: margem + (i + 1) * q, margem + j * q: margem + (j + 1) * q] = 0
    R, _ = cv2.Rodrigues(np.array([np.deg2rad(rx), np.deg2rad(ry), 0.0]))
    t = np.array([0.0, 0.0, tz])
    centro = np.array([[1, 0, -tab.shape[1] / 2], [0, 1, -tab.shape[0] / 2], [0, 0, 1]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ centro
    return cv2.warpPerspective(tab, H / H[2, 2], tamanho, borderValue=255)


@pytest.fixture(scope="module")
def pastas(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("tf")
    dados, saida = raiz / "dados", raiz / "saida"
    fotos = dados / "fotos" / "celular_teste" / "livro_teste"
    for pagina in (1, 2, 3):
        img = pagina_sintetica(pagina)
        if pagina == 3:
            # guardada deitada, com a orientação no EXIF, como faz o celular
            salvar_jpg(fotos / f"p{pagina:03d}_ref.jpg", np.rot90(img).copy(), orientacao=6)
        else:
            salvar_jpg(fotos / f"p{pagina:03d}_ref.jpg", img)
        salvar_jpg(fotos / f"p{pagina:03d}_inclinada.jpg", foto_inclinada(img, pagina))
        salvar_jpg(fotos / f"p{pagina:03d}_sombra.jpg", foto_com_sombra(img))
    salvar_jpg(fotos / "foto_sem_padrao.jpg", pagina_sintetica(9))

    K = np.array([[1200.0, 0, 800], [0, 1200.0, 600], [0, 0, 1]])
    calib = dados / "calibracao" / "celular_calibrado"
    vistas = [(0, 0), (25, 0), (-25, 0), (0, 25), (0, -25), (20, 20), (-20, 20), (20, -20)]
    for k, (rx, ry) in enumerate(vistas):
        salvar_jpg(calib / f"tab_{k:02d}.jpg", tabuleiro_visto_pela_camera(K, rx, ry, tz=900))
    return dados, saida


def rodar(script, dados, saida):
    env = {**os.environ, "TF_DADOS": str(dados), "TF_SAIDA": str(saida), "MPLBACKEND": "Agg"}
    r = subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=SCRIPTS, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_01_preparar_fotos(pastas):
    dados, saida = pastas
    saida_txt = rodar("01_preparar_fotos.py", dados, saida)
    indice = pd.read_csv(saida / "indice_fotos.csv")
    assert len(indice) == 9
    assert "foto_sem_padrao.jpg" in saida_txt
    assert set(indice["condicao"]) == {"ref", "inclinada", "sombra"}
    ref3 = indice[(indice["pagina"] == 3) & (indice["condicao"] == "ref")].iloc[0]
    assert ref3["altura"] > ref3["largura"], "a rotação do EXIF não foi corrigida"
    assert (saida / "figuras" / "exemplos_fotos.png").is_file()


def test_02_calibrar_camera(pastas):
    dados, saida = pastas
    rodar("02_calibrar_camera.py", dados, saida)
    calib = np.load(saida / "calibracao" / "celular_calibrado.npz")
    fx, fy = calib["K"][0, 0], calib["K"][1, 1]
    assert abs(fx - 1200) / 1200 < 0.05 and abs(fy - 1200) / 1200 < 0.05, (fx, fy)
    assert float(calib["erro"]) < 1.0
    assert (saida / "figuras" / "calibracao_celular_calibrado.png").is_file()


def test_03_reconhecer_pagina(pastas):
    dados, saida = pastas
    rodar("01_preparar_fotos.py", dados, saida)
    rodar("03_reconhecer_pagina.py", dados, saida)
    resultado = pd.read_csv(saida / "reconhecimento.csv")
    assert len(resultado) == 6
    assert resultado["acertou"].all(), resultado.to_string()
    assert len(list((saida / "retificadas").rglob("*.png"))) == 6
    assert (saida / "figuras" / "exemplo_reconhecimento.png").is_file()
