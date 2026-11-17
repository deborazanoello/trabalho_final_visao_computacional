# Da foto da página ao termo difícil

Trabalho final de Visão Computacional (PPGInf/UFPR). Fotos de celular de páginas de livros impressos são corrigidas, a página é reconhecida e endireitada, o texto é extraído e os termos que o leitor tende a pesquisar viram candidatos à ajuda do assistente de leitura da tese.

## Como as pastas funcionam

Igual ao notebook do Colab, mas no seu computador: os scripts leem de `dados/` e gravam em `saida/`. As duas pastas ficam fora do git.

```
dados/
  fotos/<celular>/<livro>/p012_ref.heic        foto de referência: página plana, de frente, boa luz
                         p012_inclinada.heic  fotos de uso real (o nome da condição é livre)
                         p012_sombra.heic
                         p012_curva.heic
  calibracao/<celular>/*.jpg                   fotos do tabuleiro de xadrez
saida/
  fotos_prontas/   indice_fotos.csv   calibracao/   retificadas/   reconhecimento.csv   figuras/
```

Exemplo: `dados/fotos/iphone11/memorias_postumas/p012_ref.heic`. O número da página e a condição saem do nome do arquivo, então não precisa de planilha. Aceita `.heic` (iPhone), `.jpg` e `.png`, e corrige a rotação que o celular grava na foto.

## Instalação

Python 3.10 ou mais novo.

```bash
git clone https://github.com/deborazanoello/trabalho_final_visao_computacional.git
cd trabalho_final_visao_computacional
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No VS Code, escolha o interpretador da `.venv` (Ctrl+Shift+P → *Python: Select Interpreter*).

## Rodar

Cada script tem blocos que começam com `# %%`. No VS Code, eles viram células: clique em **Run Cell** em cima de cada bloco e o resultado aparece na janela interativa, como no Colab. Também dá para rodar o arquivo inteiro:

```bash
cd scripts
python 01_preparar_fotos.py      # gira, passa para cinza, reduz e monta o índice
python 02_calibrar_camera.py     # matriz da câmera e distorção de cada celular
python 03_reconhecer_pagina.py   # qual é a página e a foto endireitada
```

As configurações ficam no topo de cada script, como `LADO_MAX` em `comum.py`, `TABULEIRO` no script 2 e `METODO` (SIFT ou ORB) e `MIN_INLIERS` no script 3.

| Script | O que faz | Tópicos da disciplina |
|---|---|---|
| `01_preparar_fotos.py` | Lê as fotos, corrige a rotação, passa para cinza e reduz com `INTER_AREA` (sem aliasing). Mede brilho, contraste e nitidez (variância do laplaciano) | Fundamentos, resample |
| `02_calibrar_camera.py` | Encontra os cantos do tabuleiro, calibra a câmera e grava a matriz e a distorção | Modelo e calibração de câmera |
| `03_reconhecer_pagina.py` | Casa a foto com as referências do livro (SIFT + teste de Lowe + RANSAC), escolhe a página com mais inliers e endireita a foto pela homografia | Características locais, transformações geométricas |

**Calibração:** imprima um tabuleiro de xadrez de 10 × 7 quadrados (ou abra na tela do computador) e tire de 15 a 20 fotos com cada celular, de ângulos diferentes, **na mesma orientação das fotos das páginas** (retrato).

## Próximos passos

4. **Limpar a imagem:** corrigir sombra e iluminação, binarizar (Otsu e Sauvola) e comparar.
5. **Extrair o texto:** OCR com o Tesseract em português (`pytesseract`; o programa Tesseract é instalado à parte).
6. **Achar os termos:** palavras raras (frequência Zipf abaixo de 3,26) e nomes próprios.
7. **Avaliar:** acerto da página, erro do OCR antes e depois de cada etapa, cobertura dos termos difíceis, por condição e por celular.
8. **Extra:** um classificador linear e um MLP para dizer se a foto está boa (a nitidez do script 1 já é uma das características).

## Testes

```bash
python -m pytest
```

Os testes criam páginas de texto sintéticas, fotos inclinadas e com sombra, uma foto com rotação no EXIF e um tabuleiro de xadrez projetado por uma câmera de parâmetros conhecidos. Depois rodam os três scripts. A calibração precisa recuperar a distância focal com erro menor que 5%, e o reconhecimento precisa acertar todas as páginas.
