# Robustez e qualidade do gate

## Dados e regras

- Protocolo principal: 12 datasets, 3 backbones e 100 seeds por par dataset/backbone (3.600 comparações pareadas). A escolha do rewiring é reconstruída exclusivamente por `val_f1`; o teste só entra após congelar a escolha.
- Bootstrap hierárquico: reamostra primeiro os datasets e, dentro de cada dataset sorteado, as execuções pareadas backbone/seed. O efeito é ganho de macro-F1 no teste em pontos percentuais, rewiring selecionado por validação menos grafo original.
- Gate: avaliação separada do gate explícito de margem de 1,0 p.p. no experimento de recuperação por corrupção (1.080 decisões). O oráculo usa o melhor teste **somente para diagnóstico** e não é parte da seleção real.

## Bootstrap hierárquico

Ganho médio: **2.577 p.p.**. IC 95% hierárquico: **[-1.591, 7.152] p.p.**.

## Controles SOTA protocol-matched (30 seeds comuns)

- proposed minus DiffWire CT: 4.616 p.p. (IC 95% [1.659, 7.630]; n=360).
- proposed minus SDRF: 2.808 p.p. (IC 95% [-1.567, 7.281]; n=1080).

## Leave-one-dataset-out

Ao retirar um dataset por vez, o ganho médio variou de **1.300** a **3.577 p.p.**. Sem os três Airports simultaneamente, o ganho médio foi **-1.371 p.p.** em 9 datasets restantes.

## Selection regret do gate de 1,0 p.p.

O regret médio foi **5.714 p.p.**; a mediana foi **4.167 p.p.**; e o gate escolheu exatamente o mesmo candidato do melhor teste em **29.8%** das decisões. O gate preservou o grafo original em **10.9%** das decisões. O ganho médio efetivamente entregue pelo gate contra o original foi **5.398 p.p.**.

## Interpretação

O bootstrap e o leave-one-dataset-out avaliam a robustez do efeito principal. O regret mede o custo de decidir com validação, em vez de conhecer antecipadamente o teste. Ele deve ser reportado como diagnóstico de seleção, sem afirmar que o oráculo é um resultado alcançável.
