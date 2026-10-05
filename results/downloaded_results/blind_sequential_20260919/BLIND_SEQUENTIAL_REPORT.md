# Validação cega sequencial

As previsões de cada onda foram congeladas antes da execução do rewiring. Somente resultados verdadeiros foram adicionados ao conhecimento.

- Grafos cegos: 96
- Acurácia: 0.646
- Balanced accuracy: 0.556
- ROC AUC: 0.430
- Precisão: 0.121
- Sensibilidade: 0.444
- Especificidade: 0.667
- Brier score: 0.216

## Evolução por onda

| Onda | n | Acurácia | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|
| blind_wave1_geometric | 32 | 0.844 | 0.482 | 0.232 |
| blind_wave2_degree_corrected | 32 | 0.906 | 0.468 | 0.290 |
| blind_wave3_hierarchical | 32 | 0.188 | 0.536 | 0.375 |

## Informações do grafo mais utilizadas pelo modelo final

- degree_cv: importância combinada 0.265
- degree_assortativity: importância combinada 0.084
- avg_degree: importância combinada 0.081
- effective_resistance_sample: importância combinada 0.070
- largest_cc_ratio: importância combinada 0.066
- avg_clustering: importância combinada 0.059
- transitivity: importância combinada 0.054
- algebraic_connectivity: importância combinada 0.051
- num_isolates: importância combinada 0.039
- modularity: importância combinada 0.038

A importância final é descritiva. A validade preditiva é medida apenas pelas previsões congeladas das ondas cegas.
