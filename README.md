# Bases de escrita à mão: trabalho final de Visão Computacional

Scripts para baixar, organizar e analisar três bases públicas de imagens de escrita à mão, para o trabalho final da disciplina de Visão Computacional (INFO7064, PPGInf/UFPR).

O tema vem da minha pesquisa sobre apoio à leitura e à escrita. O objetivo é encontrar, nas imagens, sinais de hesitação ou dificuldade na escrita, como rasuras, correções e letras espelhadas. Os rótulos descrevem **eventos na escrita** (uma palavra riscada, uma letra espelhada), e não pessoas. Nada aqui diagnostica dislexia ou disgrafia.

## Bases

| Base | O que tem | Rótulos | Licença | Link |
|---|---|---|---|---|
| **BRESSAY** | 1.000 redações manuscritas em português do Brasil, cada uma de um escritor. Imagens de página, parágrafo, linha e palavra, cada uma com a transcrição | `limpa`, `riscada`, `correcao` (texto acima ou abaixo da linha), `ilegivel`, tirados das marcas da transcrição | CC BY 4.0 | [Zenodo](https://zenodo.org/records/11637681) |
| **Dyslexia Handwriting Dataset** | Letras isoladas, em `Train/` e `Test/` | `normal`, `espelhada` (Reversal), `corrigida` (Corrected) | veja a página da base | [Kaggle](https://www.kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset) |
| **Handwritten cross-outs** | Cerca de 31 mil palavras em inglês, de 12 escritores | `clean`, 7 tipos de rasura e `mix` | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/wahlinski/handwritten_cross-outs) |

**Antes de baixar:**

- **BRESSAY:** o zip tem 1,5 GB. Reserve espaço também para a pasta extraída. Se a conexão cair, rode de novo e o download continua de onde parou.
- **Dyslexia:** o `Gambo.zip` que vem na base é **protegido por senha**. Procure a senha na página da base no Kaggle e informe com `--senha-dislexia` ou com a variável `DISLEXIA_SENHA`. O download usa o `kagglehub`. Se ele pedir login, crie um token em kaggle.com → Settings → API → *Create New Token* e salve o `kaggle.json` em `~/.kaggle/` (no Windows, `C:\Users\<você>\.kaggle\`). Se preferir, baixe pelo site e passe o arquivo com `--arquivo-dislexia`.
- **Cross-outs:** baixado pela biblioteca `datasets`, sem login.

## Instalação

Python 3.10 ou mais novo.

```bash
git clone https://github.com/deborazanoello/trabalho_final_visao_computacional.git
cd trabalho_final_visao_computacional
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Baixar as bases

```bash
python -m bases.baixar                                   # as três bases
python -m bases.baixar bressay                           # só uma
python -m bases.baixar dislexia --senha-dislexia "SENHA"
python -m bases.baixar dislexia --arquivo-dislexia ~/Downloads/archive.zip --senha-dislexia "SENHA"
```

Tudo vai para `dados/`, que fica fora do git. Rodar de novo não baixa o que já existe; use `--forcar` para refazer. Por padrão, o BRESSAY é indexado nos níveis de palavra e de linha; para outros níveis, use `--niveis-bressay palavras linhas paragrafos paginas`.

## Analisar

```bash
python -m bases.analisar                       # todas as bases baixadas
python -m bases.analisar --indices dislexia    # só uma
python -m bases.analisar --amostra 1000        # mede mais imagens por rótulo (padrão: 500)
python -m bases.analisar --sem-repetidas       # pula a busca de imagens repetidas (mais rápido)
```

O resultado vai para `relatorios/analise_bases.md`, com as figuras em `relatorios/figuras/`. Para cada base, o relatório traz:

- **Contagens:** imagens por rótulo e por partição.
- **Imagens repetidas:** arquivos idênticos, incluindo os que aparecem no treino e no teste ao mesmo tempo (vazamento) e os que têm rótulos diferentes.
- **Medidas das imagens**, numa amostra por rótulo: tamanho, intensidade média, contraste, proporção de tinta (limiar de Otsu), componentes conexos e fundo escuro.
- **Teste de atalho:** o quanto essas medidas, que não olham a forma da escrita, já separam os rótulos. Usa regressão logística e uma árvore de profundidade 3, e mostra as regras da árvore. Se medidas tão simples já separam as classes, um classificador pode aprender um atalho em vez do que interessa.
- **Exemplos** sorteados de cada rótulo.

No BRESSAY, o relatório também mostra as marcas da transcrição por partição, a proporção de palavras riscadas por página (cada página é um escritor) e as palavras mais riscadas e mais corrigidas.

## Formato dos índices

`dados/indices/<nome>.csv`, uma linha por imagem. Os nomes são `bressay_palavras`, `bressay_linhas`, `dislexia` e `rasuras_hf`.

| Coluna | Conteúdo |
|---|---|
| `base` | `bressay`, `dislexia` ou `rasuras_hf` |
| `split` | `treino`, `validacao`, `teste` (ou `sem_split`) |
| `rotulo` | rótulo principal da imagem |
| `caminho` | caminho da imagem, relativo a `dados/` |

Colunas extras: no BRESSAY, `nivel`, `pagina`, `texto`, `riscada`, `riscada_ilegivel`, `correcao` e `ilegivel`; na Dyslexia, `classe_original`; na Cross-outs, `rasurada`, `ferramenta` e `texto`.

Para usar em outro script:

```python
from bases.comum import pastas, carregar_indice, caminho_absoluto
from bases.estatisticas import ler_cinza

p = pastas()
palavras = carregar_indice(p, "bressay_palavras")
riscadas = palavras[palavras["rotulo"] == "riscada"]
img = ler_cinza(caminho_absoluto(riscadas["caminho"].iloc[0], p))  # array em tons de cinza
```

## Cuidados com os rótulos

- **BRESSAY:** os rótulos vêm das marcas da transcrição: `--texto--` é texto riscado e legível, `--xxx--` é riscado e ilegível, `##texto##` e `$$texto$$` são texto acima e abaixo da linha, e `@@???@@` é ilegível. As marcas não são exclusivas, então o rótulo principal segue a prioridade riscada > correção > ilegível > limpa. A partição oficial é por página: use-a, para que o mesmo escritor não apareça no treino e no teste.
- **Dyslexia:** as letras vêm de fontes diferentes (NIST SD19, outra base do Kaggle e crianças em Penang, na Malásia). Por isso, vale olhar o teste de atalho antes de confiar em acurácias altas. Um teste extra é espelhar letras normais e ver se o modelo passa a chamá-las de espelhadas. Use letras que não viram outra letra ao espelhar (evite b, d, p e q).
- **Cross-outs:** `mix` mistura palavras limpas e rasuradas, então a coluna `rasurada` fica vazia nesse rótulo.

## Estrutura

```
bases/
  comum.py         pastas, download, zip e índices
  bressay.py       BRESSAY
  dislexia.py      Dyslexia Handwriting Dataset
  rasuras_hf.py    Handwritten cross-outs
  baixar.py        python -m bases.baixar
  estatisticas.py  medidas das imagens, repetidas e teste de atalho
  graficos.py      gráficos da análise
  analisar.py      python -m bases.analisar
tests/             testes com mini-bases falsas (não acessam a internet)
dados/             criado pelos scripts, fora do git
relatorios/        relatório e figuras da análise
```

## Testes

```bash
python -m pytest
```

Os testes criam mini-bases com a mesma estrutura de pastas das reais, incluindo um zip com senha, e rodam o download, a indexação e a análise de ponta a ponta.

## Referências das bases

- Neto, A. F. S. et al. *BRESSAY: A Brazilian Portuguese Dataset for Offline Handwritten Text Recognition*. ICDAR 2024. Dados: https://zenodo.org/records/11637681
- *Dyslexia Handwriting Dataset*. Kaggle: https://www.kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset
- *A study of handwritten text recognition with cross-out words*. International Journal on Document Analysis and Recognition (IJDAR), 2026. doi:10.1007/s10032-026-00603-8. Dados: https://huggingface.co/datasets/wahlinski/handwritten_cross-outs
