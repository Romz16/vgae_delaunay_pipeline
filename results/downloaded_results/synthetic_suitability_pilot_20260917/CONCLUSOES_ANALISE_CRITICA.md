# Conclusões do experimento de aplicabilidade estrutural

## Síntese executiva

O experimento foi concluído sem execuções faltantes: 205 grafos independentes, 41 configurações, 6 famílias, 2.460 execuções aninhadas e 24.600 condições de GNN. Não há `test_f1` ausente nem `run_id` duplicado.

A resposta atual à pergunta principal é negativa: as propriedades estruturais do grafo original, isoladamente, não produziram um indicador confiável de quando o rewiring melhora o GNN. O modelo congelado ficou próximo do acaso nos sintéticos e não generalizou para os 12 datasets reais.

O resultado mecanístico é mais forte. A reconstrução geométrica pura foi prejudicial na maior parte dos grafos. O VGAE recuperou grande parte dessa perda, mas raramente transformou o pipeline completo em ganho líquido. Assim, o VGAE aparece principalmente como mecanismo de reparo da reconstrução geométrica, e não como garantia de que o rewiring completo será vantajoso.

## Efeito total do pipeline

As inferências abaixo usam o grafo independente como unidade científica. Seeds e regimes aninhados foram primeiro agregados dentro de cada grafo.

| Comparação | Ganho médio de macro-F1 | Mediana | IC 95% bootstrap por configuração | Proporção com ganho > 0 | Proporção com ganho > 0,5 p.p. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pipeline selecionado − original | −4,379 p.p. | −4,342 p.p. | [−5,246; −3,512] p.p. | 6,34% | 5,37% |
| Delaunay puro − original | −9,139 p.p. | −9,082 p.p. | [−10,490; −7,818] p.p. | 2,93% | 2,93% |
| Refinamento VGAE − Delaunay puro | +4,907 p.p. | +4,680 p.p. | [+4,220; +5,633] p.p. | 96,10% | 95,61% |

O efeito total selecionado é significativamente negativo no teste de Wilcoxon (`p = 3,65 × 10⁻³¹`). A conclusão não depende apenas de uma média puxada por outliers: a mediana também é negativa e o intervalo bootstrap por configuração não cruza zero.

O incremento de VGAE usa o melhor percentual positivo escolhido pela validação, mesmo quando o pipeline final escolhe `r = 0`. Por isso, ele mede se algum refinamento positivo repara o Delaunay, e não deve ser somado mecanicamente ao ganho final nas 539 execuções em que `r = 0` foi selecionado.

## Delaunay e outros construtores

Todos os construtores baseados na representação aprendida tiveram perda média antes do refinamento. Mutual-kNN e kNN foram ligeiramente melhores que Delaunay:

| Construtor sem refinamento | Ganho médio contra o original |
| --- | ---: |
| mutual-kNN | −8,682 p.p. |
| kNN | −8,890 p.p. |
| Delaunay | −9,139 p.p. |
| Random degree-matched | −38,881 p.p. |
| Random edge-budget-matched | −40,362 p.p. |

Na comparação pareada por grafo, mutual-kNN superou Delaunay em 0,457 p.p. em média (`p = 5,94 × 10⁻⁵`) e kNN em 0,249 p.p. (`p = 0,0187`). Esses testes são exploratórios e devem receber correção por multiplicidade em uma versão para publicação.

Os controles aleatórios muito piores mostram que a representação aprendida contém informação útil. Entretanto, os resultados não sustentam superioridade específica do Delaunay. A conclusão correta é que a geometria aprendida é muito melhor que um orçamento aleatório de arestas, mas nenhum dos três construtores aprendidos superou o grafo original em média.

Entre percentuais fixos de refinamento, `r = 0,40` teve a menor perda média, −4,503 p.p. O selecionador por validação chegou a −4,379 p.p., uma pequena melhora, mas ainda negativa. Os percentuais escolhidos ficaram dispersos: 21,9% em `r = 0`, 15,7% em 0,10, 19,1% em 0,25, 19,2% em 0,40 e 24,0% em 0,55. Não existe um percentual universal sustentado pelos dados.

## Regimes em que o método funcionou melhor

Random regular foi a única família com ganho médio positivo: +0,580 p.p., com 5 de 10 grafos acima de +0,5 p.p. A evidência ainda é pequena, pois essa família contém somente duas configurações. Todas as demais famílias tiveram perda média:

| Família | Ganho médio selecionado |
| --- | ---: |
| Random regular | +0,580 p.p. |
| Degree-matched bottleneck | −2,989 p.p. |
| Watts–Strogatz | −3,610 p.p. |
| SBM density-matched | −5,400 p.p. |
| Barabási–Albert | −6,397 p.p. |
| Erdős–Rényi | −7,553 p.p. |

Somente 11 de 205 grafos superaram +0,5 p.p., distribuídos por seis configurações: duas random regular, duas SBM com `mixing = 0,45` e duas Watts–Strogatz com `k = 12`. Isso sugere um regime de baixa heterogeneidade de grau e determinadas estruturas locais, mas ainda não define uma regra robusta.

A qualidade das features alterou fortemente o efeito:

| Regime de features | F1 baseline médio | Ganho selecionado médio |
| --- | ---: | ---: |
| Informativas | 0,929 | −1,682 p.p. |
| Parciais | 0,893 | −4,019 p.p. |
| Ruidosas | 0,864 | −7,436 p.p. |

A correlação entre F1 baseline e ganho foi praticamente nula (`Spearman ρ = 0,015`), portanto o resultado não se reduz a um simples efeito teto. Ainda assim, a forte diferença entre regimes mostra que uma regra exclusivamente topológica enfrenta incerteza irredutível: dois grafos com a mesma topologia podem responder de modo diferente quando a representação aprendida tem qualidade diferente.

## Interpretação estrutural

As associações brutas mais fortes com o ganho foram:

- `degree_cv`: `ρ = −0,515`;
- λ2 normalizado: `ρ = −0,407`;
- edge-connectivity ratio: `ρ = +0,401`;
- spectral sweep conductance: `ρ = −0,363`;
- average shortest path: `ρ = +0,330`.

Essas associações não permaneceram dentro de cada configuração. Depois de remover a média da configuração, as correlações ficaram próximas de zero, aproximadamente entre −0,06 e +0,08. Portanto, elas parecem principalmente separar famílias e configurações inteiras. Não há evidência de que alterar isoladamente λ2, path length ou degree CV causaria a mudança observada no ganho.

O Delaunay puro, em média:

- removeu 382 arestas e reduziu o grau médio em 3,19;
- reduziu λ2 em 0,159;
- aumentou o caminho médio em 2,15 saltos e o diâmetro em 5,12;
- aumentou clustering em 0,320 e modularidade em 0,213.

Isso descreve uma transformação para vizinhanças mais locais e agrupadas, porém com pior conectividade global. O refinamento preservou o orçamento de arestas e reduziu parte da degradação de λ2 e path length, mas o grafo selecionado ficou, em média, com 2,66 componentes adicionais. O ganho do VGAE parece vir da realocação de arestas para relações task-informative, mesmo quando a conectividade global piora.

## Avaliação do indicador estrutural

O target principal é muito desbalanceado: somente 11 positivos contra 194 negativos. O modelo congelado, baseado em λ2 e densidade, apresentou:

- ROC-AUC sintético: 0,479;
- balanced accuracy: 0,545;
- 1 verdadeiro positivo, 10 falsos negativos e nenhum falso positivo;
- leave-one-family-out ROC-AUC: 0,472, com zero verdadeiros positivos;
- Ridge para ganho contínuo: `R² = −0,323`;
- Random Forest exploratória: `R² = 0,071`.

Uma combinação exploratória de λ2 e `degree_cv` atingiu ROC-AUC 0,618, mas a política de custo escolheu um threshold que não recomendava nenhum grafo. Esse sinal é insuficiente para declarar um indicador operacional.

Há também um erro lógico na política de três regiões produzida pelo código: `avoid_threshold = 0,99` e `recommend_threshold = 0,0592`. Um threshold de Avoid maior que o de Recommend torna a região de incerteza inválida. Isso ocorreu porque o critério de Avoid permite “evitar todos” quando a prevalência positiva é inferior a 10%. O relatório automático e o indicador congelado não devem ser publicados sem corrigir esse ponto.

## Validação nos 12 datasets reais

O indicador classificou todos os 12 datasets como `Avoid`, embora 6 apresentassem ganho real acima de +0,5 p.p.:

- accuracy: 0,50;
- balanced accuracy: 0,50;
- ROC-AUC: 0,472;
- 6 verdadeiros negativos, 6 falsos negativos e nenhum verdadeiro positivo.

Somente 3 dos 12 datasets reais estavam simultaneamente dentro das faixas sintéticas de λ2 e densidade. Os sintéticos tinham 240 nós e densidade entre 0,0166 e 0,0647, enquanto vários grafos reais são muito maiores e muito mais esparsos. Assim, a validação externa falhou, mas também revelou forte mudança de domínio. Não é correto reajustar o indicador usando esses 12 datasets; eles devem permanecer como validação externa.

## Hipótese revisada

Uma hipótese compatível com os resultados é:

> O rewiring geométrico task-informed é útil quando a geometria aprendida preserva vizinhanças discriminativas sem destruir a conectividade e o orçamento estrutural necessários ao message passing. O refinamento por similaridade atua principalmente como reparo quando a projeção 2D e o construtor local removem relações úteis. A aplicabilidade depende conjuntamente da heterogeneidade de grau, do descompasso de orçamento de arestas, de gargalos/conectividade e da qualidade da representação aprendida; propriedades topológicas isoladas não bastam.

Essa hipótese preserva a ideia científica de “método geométrico + similaridade”, mas não afirma que Delaunay é o melhor construtor nem que existe hoje um threshold estrutural confiável.

## Recomendações

1. Corrigir a política de thresholds impondo `avoid_threshold < recommend_threshold`, cobertura mínima nas três regiões e calibração fora da amostra. Reexecutar somente agregação/fit/validação/relatório; os 24.600 treinamentos não precisam ser repetidos.
2. Ampliar o domínio sintético para incluir tamanhos e esparsidades dos grafos reais. Variar `n`, grau médio abaixo de 6, componentes, isolates e densidades próximas de `10⁻⁴–10⁻²`.
3. Criar intervenções pareadas sobre orçamento de arestas. Comparar substituição completa por Delaunay, Delaunay augmentado ao grafo original, Delaunay com MST/connectivity constraint e construtores com o mesmo grau médio do original.
4. Aumentar deliberadamente os regimes positivos. O conjunto atual, com 5,4% de sucessos, é insuficiente para aprender uma região de recomendação conservadora.
5. Tratar qualidade da representação como modificador de efeito. Se o indicador precisar continuar estritamente topology-only, assumir uma região `Uncertain` ampla. Como análise secundária, avaliar sinais pré-rewiring sem labels de teste, como estabilidade da representação, feature smoothness e trustworthiness do UMAP.
6. Repetir o estudo focal em random regular, SBM `mixing = 0,45` e Watts–Strogatz `k = 12`, com mais configurações e intervenções controladas. Esses são os únicos regimes que produziram os 11 sucessos relevantes.
7. Para o paper atual, apresentar o resultado como evidência condicional e mecanística. Os dados sustentam que a representação aprendida supera controles aleatórios e que VGAE repara Delaunay. Eles não sustentam um indicador estrutural pronto para uso nem superioridade do Delaunay.

## Conclusão para publicação

O piloto é cientificamente útil, mas não valida a hipótese de que métricas estruturais do grafo original já permitem decidir com segurança quando aplicar o rewiring. Ele identifica uma falha de projeto importante: a reconstrução planar local frequentemente remove conectividade útil, e o VGAE funciona como compensação parcial. A próxima contribuição publicável deve ser um mecanismo de reconstrução consciente do orçamento/conectividade, seguido de um novo experimento sintético com melhor equilíbrio de classes e maior sobreposição com grafos reais.
