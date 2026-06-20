# VGAE + DGlf/Delaunay Rewiring Pipeline

Pipeline end-to-end para experimentos de rewiring em GNNs:

1. Otimização de hiperparâmetros do VGAE via Optuna.
2. Treinamento do VGAE final e persistência dos embeddings `.npy`.
3. Otimização de GCN, GCN Residual, GraphSAGE e GAT via Optuna.
4. Construção de DGlf: GCN auxiliar -> UMAP 2D -> Delaunay.
5. Rewiring DGlf + VGAE nas taxas configuradas.
6. Avaliação final com média, desvio, CI95 e exportação para CSV/Markdown.

## Instalação

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

## Execução com dataset PyG

```bash
python3 main.py --dataset Cora --root ./data --output-dir outputs/cora
```

Exemplos de nomes suportados:

```text
Cora, Pubmed, Citeseer, Cornell, Texas, Wisconsin, Actor, WikiCS,
LastFMAsia, Flickr, Chameleon, Squirrel, Crocodile,
Airports-Brazil, Airports-USA, Airports-Europe,
Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions
```

## Execução com arquivo de grafo

```bash
python3 main.py --graph-file ./graph.pt --dataset-name MeuGrafo --output-dir outputs/meu_grafo
```

Formatos suportados:

- `.pt` / `.pth`: `torch_geometric.data.Data` ou dict com `edge_index`, `x`, `y`.
- `.npz`: arrays `edge_index`, `x`, `y`.
- `.graphml`, `.gexf`, `.gpickle`: NetworkX.
- `.csv`: edge list com colunas `source,target`.

Para classificação de nós, o grafo precisa ter labels `y`. Se não houver features `x`, o código cria features simples baseadas em grau.

## Smoke test rápido

```bash
python3 main.py \
  --dataset Wisconsin \
  --output-dir outputs/wisconsin_smoke \
  --vgae-trials 3 \
  --gnn-trials 3 \
  --vgae-opt-epochs 20 \
  --vgae-final-epochs 20 \
  --gnn-opt-epochs 20 \
  --gnn-final-epochs 20 \
  --final-runs 2
```

## Saídas

```text
outputs/
├── embeddings/<dataset>_vgae_embeddings.npy
├── results/tabela_resultados_<dataset>_100runs.csv
├── results/tabela_resultados_<dataset>_100runs.md
├── best_vgae_params.json
├── best_gnn_params.json
└── pipeline_config.json
```

## Observação metodológica

A condição `DGlf + VGAE 0%` usa o grafo DGlf/Delaunay puro. Ela não é igual à baseline original.
A baseline é sempre `Baseline (Original)`.
