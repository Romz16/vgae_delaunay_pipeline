# Conclusões do experimento de corrupção–recuperação

Foram analisados 180 grafos independentes, 1080 execuções aninhadas e 16200 condições GNN.

## Método proposto com seleção por validação

- Ganho médio: -0.802 pp.
- Mediana: -0.765 pp.
- Fração positiva: 42.2%.
- Fração acima de 0,5 pp: 36.1%.
- Delaunay sem refinamento: -2.552 pp.
- Incremento do refinamento VGAE: 2.008 pp.

## Sensibilidade do método proposto

### Corrupção

- nível 0.0: ganho -0.525 pp; sucesso 41.7%.
- nível 0.25: ganho -1.200 pp; sucesso 29.2%.
- nível 0.5: ganho -0.544 pp; sucesso 40.3%.

### Homofilia latente

- nível 0.2: ganho -0.419 pp; sucesso 46.7%.
- nível 0.5: ganho -0.628 pp; sucesso 31.7%.
- nível 0.8: ganho -1.360 pp; sucesso 30.0%.

## Métricas candidatas

| Métrica | Spearman com ganho | AUC orientada | Direção |
|---|---:|---:|---|
| observed_edge_homophily | -0.102 | 0.589 | lower predicts success |
| degree_cv | -0.033 | 0.565 | lower predicts success |
| baseline_val_f1 | -0.100 | 0.559 | lower predicts success |
| algebraic_connectivity | 0.016 | 0.556 | higher predicts success |
| lambda2_norm_laplacian | 0.007 | 0.549 | higher predicts success |
| effective_resistance_sample | -0.052 | 0.541 | lower predicts success |
| avg_degree | 0.054 | 0.539 | higher predicts success |
| degree_assortativity | -0.015 | 0.536 | lower predicts success |

## Validação fora da amostra por configuração

- Modelo multivariado: AUC 0.506; balanced accuracy 0.483.
- Árvore rasa: AUC 0.514; balanced accuracy 0.493.
- Somente topologia, por tarefa: AUC 0.445; balanced accuracy 0.449.
- Topologia + homofilia, por tarefa: AUC 0.442; balanced accuracy 0.450.
- Somente F1 de validação do baseline, por tarefa: AUC 0.518; balanced accuracy 0.518.
- Modelo operacional completo, por tarefa: AUC 0.463; balanced accuracy 0.482.

## Faixas do F1 de validação do baseline

| Faixa | n | Ganho médio (pp) | Sucesso >0,5 pp |
|---|---:|---:|---:|
| 0.40–0.55 | 39 | 0.609 | 51.3% |
| 0.55–0.70 | 365 | -0.529 | 42.2% |
| >0.70 | 136 | -1.941 | 36.8% |

## Gate de validação entre construtores

| Margem mínima na validação (pp) | Execuções com rewiring | Ganho médio no teste (pp) | IC95% |
|---:|---:|---:|---:|
| 0.0 | 91.9% | 5.473 | [4.928, 6.068] |
| 0.5 | 90.5% | 5.421 | [4.862, 6.001] |
| 1.0 | 89.1% | 5.398 | [4.833, 5.971] |
| 2.0 | 86.7% | 5.325 | [4.760, 5.880] |
| 3.0 | 82.4% | 5.201 | [4.691, 5.744] |

A árvore abaixo é apenas descritiva no conjunto completo; não deve ser usada como regra confirmatória sem validação externa.

```
|--- observed_edge_homophily <= 0.3234
|   |--- learned_umap_trustworthiness <= 0.8416
|   |   |--- class: 1
|   |--- learned_umap_trustworthiness >  0.8416
|   |   |--- class: 1
|--- observed_edge_homophily >  0.3234
|   |--- degree_cv <= 0.3988
|   |   |--- class: 1
|   |--- degree_cv >  0.3988
|   |   |--- class: 0
```
