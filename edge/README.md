# OpsSentinel - MVP de deteccao de EPI (capacete e colete) na borda

Protótipo interno, 100% local (sem nuvem), para detectar via webcam pessoas
**sem capacete** e/ou **sem colete de seguranca**, usando YOLO (Ultralytics).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Usamos Python 3.12 explicitamente porque o PyTorch (dependencia do
> Ultralytics) ainda nao tem wheels estaveis para versoes mais novas do
> Python neste momento.

## Dataset

Kaggle: [`shlokraval/ppe-dataset-yolov8`](https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8)
(export Roboflow v4, 640x640 stretch, 44.002 imagens).

Classes de interesse (do data.yaml ORIGINAL, nc=14, por indice):
`3=Hardhat`, `8=NO-Hardhat`, `10=NO-Safety Vest`, `11=Person`, `13=Safety Vest`.

**Estrategia**: nao reescrevemos rotulos. Baixamos o dataset completo uma
vez e filtramos as classes no treino via parametro `classes=[3,8,10,11,13]`
do Ultralytics.

### Credenciais do Kaggle (necessarias para a Etapa 1)

O script nunca grava nem pede sua chave em texto. Configure **uma** das
opcoes abaixo antes de rodar `src/download_dataset.py`:

1. Baixe `kaggle.json` em https://www.kaggle.com/settings ("Create New
   Token") e salve em `~/.kaggle/kaggle.json`, depois `chmod 600` nesse
   arquivo.
2. Ou exporte variaveis de ambiente na sessao do terminal:
   ```bash
   export KAGGLE_USERNAME="seu_usuario"
   export KAGGLE_KEY="sua_chave"
   ```

Nunca commitar `kaggle.json` nem essas variaveis. `.gitignore` ja exclui
`~/.kaggle`, `kaggle.json`, `data/` e `*.env`.

## Etapas

1. **Download e inspecao** (`src/download_dataset.py`, `src/inspect_dataset.py`)
2. **Treino** (`src/train.py`) - a vir
3. **Inferencia na webcam** (`src/webcam_infer.py`) - a vir
4. **Avaliacao de performance** (`src/benchmark.py`) - a vir

## Pendencias / decisoes registradas (nao resolvidas automaticamente)

- **Licenca conflitante do dataset**: o Kaggle lista Apache 2.0, mas a
  origem (Roboflow) indica CC BY 4.0. Ate esclarecer com a fonte original,
  este dataset e usado **apenas para uso interno/prototipo**, nunca para
  redistribuicao ou produto comercial.
- **Licenca do Ultralytics (AGPL-3.0)**: o pacote `ultralytics` e licenciado
  sob AGPL-3.0. Uso comercial/producao pode exigir licenca comercial paga
  da Ultralytics ou substituicao por outra implementacao de YOLO com
  licenca permissiva. Decisao de negocio pendente - nao resolvida aqui.
- **Caminhos relativos do `data.yaml` original**: o arquivo original do
  dataset usa caminhos relativos (`../train/images`) que podem quebrar
  dependendo de onde o dataset e extraido. Por isso geramos um
  `data_local.yaml` com caminhos absolutos (ver `src/download_dataset.py`).
- **Contagem de anotacoes por classe**: nao havia contagem verificada antes
  deste projeto. `src/inspect_dataset.py` gera essa contagem e alerta sobre
  desbalanceamento, mas a decisao de como tratar desbalanceamento (ex.:
  oversampling, pesos de classe, mais dados) fica para depois de ver os
  numeros reais.
