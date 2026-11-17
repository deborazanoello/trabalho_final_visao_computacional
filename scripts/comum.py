"""Pastas e funções usadas por todos os scripts.

As pastas ficam na raiz do repositório:
    dados/   fotos originais (fora do git)
    saida/   tudo o que os scripts geram (fora do git)

Para usar outras pastas, defina as variáveis de ambiente TF_DADOS e TF_SAIDA.
"""

import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()  # abre as fotos .heic do iPhone
except ImportError:
    pillow_heif = None

# ===== Pastas =====
RAIZ = Path(__file__).resolve().parent.parent
DADOS = Path(os.environ.get("TF_DADOS", RAIZ / "dados"))
SAIDA = Path(os.environ.get("TF_SAIDA", RAIZ / "saida"))

PASTA_FOTOS = DADOS / "fotos"            # fotos/<celular>/<livro>/p012_ref.heic
PASTA_CALIBRACAO = DADOS / "calibracao"  # calibracao/<celular>/*.jpg
PASTA_PRONTAS = SAIDA / "fotos_prontas"
PASTA_FIGURAS = SAIDA / "figuras"
INDICE_FOTOS = SAIDA / "indice_fotos.csv"

# ===== Padrões =====
LADO_MAX = 2000  # lado maior das fotos depois de reduzidas, em pixels
EXTENSOES = (".heic", ".heif", ".jpg", ".jpeg", ".png")
NOME_FOTO = re.compile(r"^p(\d+)_([a-z0-9]+)$", re.IGNORECASE)  # p012_ref, p012_sombra


# ===== Ler e gravar imagens =====
def ler_foto_cinza(caminho):
    """Abre a foto (HEIC, JPG ou PNG), corrige a rotação do celular e passa para cinza."""
    img = ImageOps.exif_transpose(Image.open(caminho))
    return np.array(img.convert("L"))


def reduzir(cinza, lado_max=LADO_MAX):
    """Reduz a imagem até o lado maior ter lado_max pixels. INTER_AREA evita aliasing."""
    h, w = cinza.shape
    escala = lado_max / max(h, w)
    if escala >= 1:
        return cinza
    return cv2.resize(cinza, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA)


def ler_png(caminho):
    """cv2.imread que funciona com acentos no caminho (Windows)."""
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"não consegui ler {caminho}")
    return img


def salvar_png(caminho, img):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"não consegui gravar {caminho}")
    buffer.tofile(str(caminho))


# ===== Nome das fotos =====
def info_do_nome(caminho):
    """Lê celular, livro, página e condição do caminho fotos/<celular>/<livro>/p012_cond.ext.

    Devolve None quando o caminho não segue esse padrão.
    """
    try:
        partes = Path(caminho).relative_to(PASTA_FOTOS).parts
    except ValueError:
        return None
    if len(partes) != 3:
        return None
    celular, livro, arquivo = partes
    achou = NOME_FOTO.match(Path(arquivo).stem)
    if not achou:
        return None
    return {"celular": celular, "livro": livro, "pagina": int(achou.group(1)), "condicao": achou.group(2).lower()}


# ===== Calibração =====
def arquivo_calibracao(celular):
    return SAIDA / "calibracao" / f"{celular}.npz"


def carregar_calibracao(celular):
    """Matriz da câmera, distorção e tamanho da imagem, ou None se o celular não foi calibrado."""
    arquivo = arquivo_calibracao(celular)
    if not arquivo.exists():
        return None
    d = np.load(arquivo)
    return d["K"], d["dist"], tuple(int(x) for x in d["tamanho"])


def corrigir_distorcao(img, calibracao):
    """Remove a distorção da lente. A foto precisa ter o tamanho das fotos de calibração."""
    K, dist, tamanho = calibracao
    if (img.shape[1], img.shape[0]) != tamanho:
        print(f"  aviso: foto {img.shape[1]}x{img.shape[0]} e calibração {tamanho[0]}x{tamanho[1]}; "
              "distorção não corrigida (fotografe a calibração na mesma orientação)")
        return img
    novo_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, tamanho, alpha=0)
    return cv2.undistort(img, K, dist, None, novo_K)


# ===== Figuras =====
def mostrar(fig, nome):
    """Salva a figura em saida/figuras e mostra na tela (ou na janela interativa do VS Code)."""
    import matplotlib
    import matplotlib.pyplot as plt

    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_FIGURAS / nome, dpi=150, bbox_inches="tight")  # savefig ANTES do show
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()
