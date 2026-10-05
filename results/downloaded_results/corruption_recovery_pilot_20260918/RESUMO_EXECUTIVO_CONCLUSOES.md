# Resumo executivo do experimento de corrupção–recuperação

## Escopo

O experimento contém 180 grafos independentes, 1.080 execuções aninhadas e 16.200 condições GNN. Os rótulos foram gerados antes do grafo observado, evitando que a tarefa de classificação fosse definida pela própria topologia avaliada. A análise estatística usa o grafo independente como unidade e mantém configurações relacionadas juntas na validação cruzada.

## Resultado principal

O pipeline fixo auxiliary GCN → UMAP → Delaunay → seleção do percentual VGAE não apresentou benefício médio neste experimento:

- ganho médio com percentual selecionado por validação: -0,802 ponto percentual;
- mediana: -0,765 ponto percentual;
- 42,2% dos grafos tiveram ganho positivo;
- 36,1% superaram 0,5 ponto percentual;
- Delaunay sem refinamento: -2,552 pontos percentuais;
- o refinamento VGAE recuperou em média 2,008 pontos percentuais, mas não compensou toda a perda causada pela reconstrução geométrica.

Percentuais maiores, 40% e 55%, reduziram a perda do Delaunay, mas não produziram ganho médio estatisticamente positivo. A união entre o Delaunay e o grafo original também foi negativa, com -1,091 ponto percentual.

## O que funcionou

Os construtores baseados diretamente nas features brutas foram claramente superiores:

- kNN em alta dimensão sobre features brutas: +6,910 pontos percentuais, IC95% [6,353; 7,512], positivo em 93,9% dos grafos;
- UMAP das features brutas seguido de Delaunay: +4,420 pontos percentuais, IC95% [3,836; 5,027], positivo em 86,7% dos grafos;
- kNN em alta dimensão sobre a representação aprendida: +0,550 ponto percentual, IC95% [0,027; 1,070], mas sem significância após correção de Holm.

O resultado depende fortemente da qualidade das features. O kNN bruto ganhou +14,462 pontos percentuais no regime informativo, +6,086 no parcial e apenas +0,182 no ruidoso. O Delaunay sobre UMAP das features brutas ganhou +12,243, +1,325 e -0,307 pontos percentuais, respectivamente.

Isso indica que a representação bruta já continha o sinal útil para reconstruir vizinhanças. O auxiliary GCN, treinado sobre o grafo corrompido, incorporou parte dos defeitos da topologia observada e enfraqueceu esse sinal antes do UMAP e do Delaunay.

## Existe uma métrica que diga quando usar?

Não foi encontrada uma métrica estrutural isolada suficientemente confiável.

- homofilia observada foi o melhor indicador individual, mas sua AUC foi apenas 0,589 e a correlação com o ganho foi -0,102;
- degree CV teve AUC 0,565;
- conectividade algébrica teve AUC 0,556;
- lambda2 teve AUC 0,549;
- resistência efetiva teve AUC 0,541;
- densidade, clustering, modularidade, caminhos e trustworthiness ficaram próximos do acaso.

Na validação cruzada mantendo configurações inteiras fora do treino:

- métricas puramente topológicas: AUC 0,445 e balanced accuracy 0,449;
- topologia mais homofilia: AUC 0,442 e balanced accuracy 0,450;
- F1 de validação do baseline isolado: AUC 0,518;
- conjunto operacional completo, incluindo trustworthiness: AUC 0,463.

Portanto, os limiares encontrados por uma árvore no conjunto completo não devem ser usados como regra. Eles não se sustentaram fora da amostra.

## Regra operacional recomendada

A evidência favorece um gate de validação direta entre construtores, não uma regra baseada em uma métrica estrutural fixa.

Procedimento recomendado:

1. Treinar o baseline no grafo original.
2. Construir somente os candidatos de custo aceitável: original, raw-feature kNN, raw-feature UMAP → Delaunay, learned-feature kNN e Delaunay com os percentuais candidatos.
3. Avaliar todos com os mesmos splits e seeds.
4. Aplicar rewiring apenas se o melhor candidato superar o original em pelo menos 1 ponto percentual na validação, idealmente de forma consistente em mais de um split.
5. Caso contrário, manter o grafo original.

Com margem de 1 ponto percentual na validação, esse gate obteve:

- ganho médio no teste de +5,398 pontos percentuais;
- IC95% [4,833; 5,971];
- ganho positivo em 93,9% dos grafos;
- ganho superior a 0,5 ponto percentual em 89,4% dos grafos.

Esse número é exploratório porque a margem foi examinada no mesmo estudo. Deve ser pré-registrada e confirmada em novos grafos e nos datasets reais.

## Interpretação para a hipótese

Os resultados não apoiam “Delaunay é superior” nem “métricas estruturais simples identificam previamente os grafos adequados”. Eles apoiam uma hipótese mais específica:

> O rewiring é útil quando a representação usada para construir as vizinhanças preserva sinal discriminativo que a topologia original não expressa bem. O construtor e a intensidade do rewiring devem ser escolhidos por validação; quando a representação aprendida herda os defeitos do grafo observado, a reconstrução pode piorar o resultado.

O componente geométrico continua relevante porque raw-feature UMAP → Delaunay foi fortemente positivo. Entretanto, a etapa crítica parece ser a escolha da representação, seguida pela validação do construtor, e não o Delaunay isoladamente.

## Próximos passos

1. Repetir o gate com validação cruzada aninhada e margem pré-fixada de 1 ou 2 pontos percentuais.
2. Aplicar o mesmo gate aos 12 datasets reais, sem alterar splits, seeds ou orçamento dos demais métodos.
3. Medir separabilidade das features antes do rewiring, por exemplo kNN validation score, margem entre classes e razão de distâncias intra/interclasse. Essas métricas são candidatas mais coerentes que lambda2 ou densidade.
4. Comparar auxiliary GCN treinado no grafo observado com um encoder independente da topologia ou robusto à corrupção.
5. Reportar tempo e custo de cada construtor. O raw-feature kNN é um candidato especialmente forte por evitar auxiliary GCN, UMAP e VGAE.
