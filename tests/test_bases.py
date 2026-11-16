"""Testes com mini-bases falsas que imitam a estrutura de pastas das bases reais.

Nenhum teste acessa a internet. Rode com:  python -m pytest
"""

from __future__ import annotations

import io
import zipfile

import cv2
import numpy as np
import pandas as pd
import pyzipper
import pytest

from bases import analisar, baixar, bressay, dislexia, rasuras_hf
from bases.comum import SenhaNecessaria, carregar_indice, pastas

SENHA = "segredo"


def _imagem(largura=48, altura=24, riscar=False, fundo_escuro=False, semente=0) -> np.ndarray:
    rng = np.random.default_rng(semente)
    img = np.full((altura, largura), 245, np.uint8)
    for _ in range(4):
        x1, x2 = sorted(rng.integers(2, largura - 2, 2))
        y1, y2 = rng.integers(3, altura - 3, 2)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 20, 2)
    if riscar:
        cv2.line(img, (0, altura // 2), (largura - 1, altura // 2), 10, 3)
    return 255 - img if fundo_escuro else img


def _png(img) -> bytes:
    ok, dados = cv2.imencode(".png", img)
    assert ok
    return dados.tobytes()


# ---------------------------------------------------------------- BRESSAY falso

TEXTOS = ["casa", "--erro--", "##certo##", "@@???@@", "--xxx--", "Palavra", "--erro--", "##--de--##"]
PAGINAS = {"treino": ["0001-001", "0002-001", "0003-001"], "validacao": ["0004-001"], "teste": ["0005-001"]}


def _criar_zip_bressay(p):
    arquivos = {}
    todas = [pg for lista in PAGINAS.values() for pg in lista]
    for pagina in todas:
        for k, texto in enumerate(TEXTOS):
            base = f"bressay/data/words/{pagina}/{pagina}-{k // 4:02d}/{k:03d}"
            arquivos[base + ".png"] = _png(_imagem(riscar=texto.startswith("--"), semente=k))
            arquivos[base + ".txt"] = texto.encode()
        linha = f"bressay/data/lines/{pagina}/{pagina}-01"
        arquivos[linha + ".png"] = _png(_imagem(largura=200, altura=30))
        arquivos[linha + ".txt"] = " ".join(TEXTOS).encode()
    nomes_sets = {"treino": "training", "validacao": "validation", "teste": "test"}
    for split, lista in PAGINAS.items():
        arquivos[f"bressay/sets/{nomes_sets[split]}.txt"] = ("\n".join(lista) + "\n").encode()
    destino = p.brutos / "bressay" / "bressay.zip"
    destino.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destino, "w") as z:
        for nome, dados in arquivos.items():
            z.writestr(nome, dados)


# ---------------------------------------------------------------- Kaggle falso


def _criar_zip_dislexia(pasta):
    """archive.zip (sem senha) com Gambo.zip (com senha) dentro, como no Kaggle."""
    interno = io.BytesIO()
    with pyzipper.AESZipFile(interno, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as z:
        z.setpassword(SENHA.encode())
        for split in ("Train", "Test"):
            for classe, fundo in (("Normal", False), ("Reversal", True), ("Corrected", True)):
                for i in range(8):
                    img = _imagem(32, 32, fundo_escuro=fundo, semente=i)
                    z.writestr(f"Gambo/{split}/{classe}/{classe.lower()}_{i}.png", _png(img))
    externo = pasta / "archive.zip"
    with zipfile.ZipFile(externo, "w") as z:
        z.writestr("Gambo.zip", interno.getvalue())
    return externo


# ---------------------------------------------------------------- Hugging Face falso


def _hf_falso():
    def exemplos(n):
        rotulos = ["clean", "single-line", "scratch", "mix"]
        return [
            {
                "image": {"bytes": _png(_imagem(riscar=i % 4 in (1, 2), semente=i)), "path": None},
                "label": rotulos[i % 4],
                "writing_tool": "pencil" if i % 2 else "ballpoint",
                "text": f"w{i}",
            }
            for i in range(n)
        ]

    return {"train": exemplos(40), "validation": exemplos(8), "test": exemplos(8)}


@pytest.fixture
def p(tmp_path):
    return pastas(tmp_path / "dados")


# ---------------------------------------------------------------- testes


def test_classificar_texto():
    assert bressay.classificar_texto("casa")["rotulo"] == "limpa"
    assert bressay.classificar_texto("--erro--")["rotulo"] == "riscada"
    assert bressay.classificar_texto("--xxx--")["riscada_ilegivel"]
    assert bressay.classificar_texto("##certo##")["rotulo"] == "correcao"
    assert bressay.classificar_texto("$$certo$$")["rotulo"] == "correcao"
    assert bressay.classificar_texto("@@???@@")["rotulo"] == "ilegivel"
    inserida_riscada = bressay.classificar_texto("##--de--##")
    assert inserida_riscada["riscada"] and inserida_riscada["correcao"]
    assert bressay.classificar_texto("bem-estar")["rotulo"] == "limpa"


def test_bressay_extrai_e_indexa_sem_baixar(p):
    _criar_zip_bressay(p)  # com o zip presente, nada é baixado
    raiz = bressay.preparar(p)
    df = bressay.indexar(raiz, p, "palavras")
    assert len(df) == 5 * len(TEXTOS)
    assert df.groupby("split")["pagina"].nunique().to_dict() == {"teste": 1, "treino": 3, "validacao": 1}
    assert df["rotulo"].value_counts().to_dict() == {"riscada": 20, "correcao": 5, "limpa": 10, "ilegivel": 5}
    assert (p.dados / df["caminho"].iloc[0]).is_file()


def test_dislexia_pede_senha_e_depois_indexa(p, tmp_path):
    archive = _criar_zip_dislexia(tmp_path)
    with pytest.raises(SenhaNecessaria):
        dislexia.preparar(p, arquivo=archive)
    with pytest.raises(RuntimeError, match="senha incorreta"):
        dislexia.preparar(p, senha="errada")
    pasta = dislexia.preparar(p, senha=SENHA)
    df = dislexia.indexar(pasta, p)
    assert len(df) == 48
    assert set(df["rotulo"]) == {"normal", "espelhada", "corrigida"}
    assert df["split"].value_counts().to_dict() == {"treino": 24, "teste": 24}


def test_rasuras_hf_grava_imagens(p, monkeypatch):
    monkeypatch.setattr(rasuras_hf, "_carregar_hf", _hf_falso)
    df = rasuras_hf.preparar_e_indexar(p)
    assert len(df) == 56
    assert set(df["split"]) == {"treino", "validacao", "teste"}
    assert df.loc[df["rotulo"] == "clean", "rasurada"].eq(False).all()
    assert df.loc[df["rotulo"] == "mix", "rasurada"].isna().all()
    assert all((p.dados / c).is_file() for c in df["caminho"])


def test_linha_de_comando_completa(p, tmp_path, monkeypatch):
    _criar_zip_bressay(p)
    archive = _criar_zip_dislexia(tmp_path)
    monkeypatch.setattr(rasuras_hf, "_carregar_hf", _hf_falso)

    codigo = baixar.main(
        ["--dados", str(p.dados), "--senha-dislexia", SENHA, "--arquivo-dislexia", str(archive)]
    )
    assert codigo == 0
    for nome in ("bressay_palavras", "bressay_linhas", "dislexia", "rasuras_hf"):
        assert isinstance(carregar_indice(p, nome), pd.DataFrame)

    saida = tmp_path / "relatorios"
    assert analisar.main(["--dados", str(p.dados), "--saida", str(saida), "--amostra", "20", "--exemplos", "3"]) == 0
    relatorio = (saida / "analise_bases.md").read_text(encoding="utf-8")
    for trecho in ("BRESSAY (palavras)", "Dyslexia Handwriting", "Handwritten cross-outs", "Teste de atalho", "erro"):
        assert trecho in relatorio
    figuras = {f.name for f in (saida / "figuras").glob("*.png")}
    assert {"bressay_palavras_contagem.png", "dislexia_medidas.png", "rasuras_hf_exemplos.png"} <= figuras


def test_texto_na_continua_texto(p):
    df = pd.DataFrame({"base": ["x"], "split": ["treino"], "rotulo": ["limpa"], "caminho": ["a.png"], "texto": ["NA"]})
    from bases.comum import salvar_indice

    salvar_indice(df, p, "teste_na")
    assert carregar_indice(p, "teste_na")["texto"].iloc[0] == "NA"
