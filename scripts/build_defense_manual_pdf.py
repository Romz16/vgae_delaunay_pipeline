#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_complete_research_booklet_pdf as base


ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "output" / "pdf" / "Manual_Detalhado_Defesa_Rewiring_Geometrico_Hibrido_20260924.pdf"
REAL = ROOT / "outputs_revision" / "downloaded_results" / "global_12datasets_corrected_20260824"
SOTA = ROOT / "outputs_revision" / "downloaded_results" / "global_protocol_matched_sota_controls_N30_lean"
SYN = ROOT / "outputs_revision" / "downloaded_results" / "synthetic_suitability_pilot_20260917"
REC = ROOT / "outputs_revision" / "downloaded_results" / "corruption_recovery_pilot_20260918"
BLIND = ROOT / "outputs_revision" / "downloaded_results" / "blind_sequential_20260919"
NESTED = ROOT / "outputs_revision" / "downloaded_results" / "nested_validation_auxiliary_pilot_20260920"


def qbox(question: str, answer: str, st):
    data = [[Paragraph(f"<b>Pergunta:</b> {question}", st["body"])], [Paragraph(f"<b>Resposta curta:</b> {answer}", st["body"])]]
    table = Table(data, colWidths=[16.6 * cm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), base.PALE_BLUE),
        ("BACKGROUND", (0, 1), (0, 1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, base.BLUE),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, base.GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def formula_box(title: str, formula: str, explanation: str, st):
    centered = ParagraphStyle("FormulaCentered", parent=st["body"], alignment=TA_CENTER, fontName="Arial-Bold", fontSize=11, leading=15)
    rows = [
        [Paragraph(title, st["table_head"])],
        [Paragraph(formula, centered)],
        [Paragraph(explanation, st["note"])],
    ]
    table = Table(rows, colWidths=[16.6 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), base.NAVY),
        ("BACKGROUND", (0, 1), (0, 1), base.PALE_GREEN),
        ("BOX", (0, 0), (-1, -1), 0.8, base.TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def load_extra():
    real = pd.read_csv(REAL / "final_selected_test_results.csv")
    paired = pd.read_csv(REAL / "paired_statistical_tests.csv")
    corr = pd.read_csv(REAL / "structural_metric_correlations.csv")
    constructors = pd.read_csv(SOTA / "constructor_summary.csv")
    sota = pd.read_csv(SOTA / "sota_summary.csv")
    family = pd.read_csv(SYN / "analysis_family_summary.csv")
    gate = pd.read_csv(REC / "validation_gate_1pp_reproduction_summary.csv")
    blind = pd.read_csv(BLIND / "blind_all_frozen_predictions_scored.csv")
    nested = pd.read_csv(NESTED / "nested_auxiliary_summary.csv")
    selected = pd.read_csv(NESTED / "nested_auxiliary_selected_per_seed.csv")
    return real, paired, corr, constructors, sota, family, gate, blind, nested, selected


def cover(st):
    rows = [
        ["Finalidade", "Preparação completa para apresentação, defesa e resposta a dúvidas"],
        ["Conteúdo", "Metodologia, métricas, resultados, decisões, limitações, roteiro e perguntas de banca"],
        ["Evidência", "Resultados por seed, controles de construtor, SOTA, sintéticos, validação cega e piloto final"],
        ["Atualização", "24 de setembro de 2026"],
    ]
    return [
        Spacer(1, 1.4 * cm),
        HRFlowable(width="100%", thickness=6, color=base.TEAL),
        Spacer(1, 0.7 * cm),
        base.p("Manual detalhado para apresentação e defesa", st["cover_title"]),
        base.p("Rewiring geométrico e híbrido em redes neurais de grafos", st["cover_sub"]),
        Spacer(1, 0.55 * cm),
        base.callout(
            "Este documento foi escrito para permitir que o leitor explique o estudo desde os fundamentos, interprete cada resultado e responda às principais críticas metodológicas sem depender do código durante a apresentação.",
            st["callout"], base.PALE_GREEN, base.TEAL,
        ),
        Spacer(1, 0.8 * cm),
        base.data_table(["Item", "Descrição"], rows, [3.1 * cm, 13.5 * cm], st, font_size=8.5),
        Spacer(1, 1.0 * cm),
        base.p("Leitura recomendada: use os capítulos 1 a 20 para compreender a pesquisa e os capítulos 21 a 37 como material de estudo para a defesa.", st["small"]),
        PageBreak(),
    ]


def add_extended_manual(st, extra):
    real, paired, corr, constructors, sota, family, gate, blind, nested, selected = extra
    story = []

    story += [PageBreak(), base.heading("21. Como estudar e apresentar este trabalho", st["h1"], 0)]
    story.append(base.p("A apresentação fica mais clara quando separa quatro níveis. Primeiro, o problema: o grafo observado pode ter conexões inadequadas para a tarefa. Segundo, a proposta: aprender uma representação e reconstruir vizinhanças. Terceiro, a evidência: medir ganhos sob protocolos pareados e controles. Quarto, a conclusão: o efeito existe, mas depende de representação, construtor, backbone e seleção por validação.", st["body"]))
    levels = [
        ["Nível", "Pergunta que você deve responder", "Mensagem principal"],
        ["Problema", "Por que mudar as arestas?", "A GNN mistura informações segundo a topologia; arestas ruins podem prejudicar."],
        ["Método", "O que o pipeline faz?", "Cria coordenadas informativas, propõe vizinhos e refina parte das relações."],
        ["Validação", "Como evitar escolher pelo teste?", "Escolhe checkpoint e topologia em validações separadas; o teste fica fechado."],
        ["Evidência", "Onde funcionou?", "Houve ganhos fortes em combinações reais e ganhos limpos no piloto final em Roman-Empire."],
        ["Limite", "É universal?", "Não. O próprio método usa validação para recusar rewiring quando não ajuda."],
    ]
    story.append(base.data_table(levels[0], levels[1:], [2.7 * cm, 5.0 * cm, 8.9 * cm], st, font_size=8.0, header_color=base.GREEN))
    story.append(base.heading("21.1 A resposta de 30 segundos", st["h2"], 1))
    story.append(base.callout("A pesquisa testa se uma GNN pode melhorar quando reconstruímos seu grafo a partir de uma representação aprendida. Os experimentos mostram ganhos relevantes em alguns datasets, mas não em todos. O benefício não pertence somente ao Delaunay nem ao classificador auxiliar: ele surge da interação entre representação, construção de vizinhança, refinamento, backbone e seleção por validação.", st["callout"]))
    story.append(base.heading("21.2 A resposta de dois minutos", st["h2"], 1))
    story.append(base.p("O método recebe o grafo e suas features. Um Auxiliary GCN aprende uma representação dos nós usando apenas os rótulos permitidos pelo treino. O UMAP reduz essa representação para duas dimensões. Um construtor, como Delaunay ou kNN, cria uma topologia candidata. Um VGAE pode recuperar relações por similaridade. Por fim, GCN, GAT ou GraphSAGE é treinado nessa topologia. A escolha entre o grafo original e os candidatos usa validação, nunca o teste. A investigação começou com 12 datasets reais, acrescentou métricas estruturais, controles de densidade, outros construtores, SDRF e DiffWire, estudos sintéticos, validação cega e um piloto com validações separadas. O resultado final é uma hipótese condicional, não uma promessa universal.", st["body"]))

    story += [PageBreak(), base.heading("22. Fundamentos: por que a topologia importa", st["h1"], 0)]
    story.append(base.heading("22.1 O que uma GNN faz", st["h2"], 1))
    story.append(base.p("Uma GNN atualiza a representação de cada nó agregando informação de seus vizinhos. Em linguagem simples, cada nó pergunta aos nós conectados: 'o que vocês sabem?'. Depois combina essas respostas com suas próprias features. Repetir esse processo em várias camadas permite incorporar informação de vizinhos mais distantes.", st["body"]))
    story.append(formula_box("Atualização simplificada de um nó", "h_i^(l+1) = combinação(h_i^l, agregação dos vizinhos de i)", "h_i é a representação do nó i. A lista de vizinhos vem diretamente das arestas. Portanto, mudar arestas muda o fluxo de informação.", st))
    story.append(base.heading("22.2 Três problemas possíveis no grafo observado", st["h2"], 1))
    story += base.bullets([
        "Arestas ausentes: nós que deveriam trocar informação não estão conectados.",
        "Arestas ruidosas: nós sem relação útil estão conectados e misturam sinais incompatíveis.",
        "Topologia desalinhada com a tarefa: a relação que gerou o grafo não é a mesma relação necessária para prever os rótulos.",
    ], st["bullet"])
    story.append(base.heading("22.3 Homofilia e heterofilia", st["h2"], 1))
    story.append(base.p("Homofilia significa que nós da mesma classe tendem a estar conectados. Heterofilia significa que classes diferentes se conectam com frequência. Uma GNN simples costuma se beneficiar de homofilia, mas isso não significa que toda aresta entre classes diferentes seja ruim. Em alguns domínios, a diferença entre vizinhos é justamente a informação relevante. Por isso a homofilia é um indicador, não uma regra universal.", st["body"]))
    story.append(qbox("Se aumentar a homofilia, o modelo sempre melhora?", "Não. É possível aumentar a homofilia e ao mesmo tempo fragmentar o grafo, criar atalhos artificiais ou remover relações úteis. A métrica deve ser interpretada junto com conectividade, densidade e desempenho de validação.", st))
    story.append(base.heading("22.4 Oversmoothing e oversquashing", st["h2"], 1))
    story.append(base.p("Oversmoothing ocorre quando muitas agregações deixam os nós excessivamente parecidos. Oversquashing ocorre quando muita informação precisa atravessar poucos caminhos e é comprimida em vetores pequenos. Rewiring pode ajudar criando rotas úteis, mas também pode piorar se aumentar demais a mistura ou criar gargalos. Esse é um dos motivos para medir lambda2, Cheeger, diâmetro e resistência efetiva.", st["body"]))

    story += [PageBreak(), base.heading("23. O pipeline explicado etapa por etapa", st["h1"], 0)]
    stages = [
        ["Etapa", "Entrada", "Saída", "Por que existe", "Falha possível"],
        ["1. Grafo original", "Features e arestas", "Baseline", "Define a referência", "O grafo já pode ser bom; mudar pode piorar"],
        ["2. Auxiliary GCN", "Treino e grafo permitido", "Embeddings", "Aprender relações úteis à tarefa", "Copiar ruído do grafo ou sobreajustar"],
        ["3. UMAP", "Embeddings", "Coordenadas 2D", "Permitir construção geométrica", "Distorcer vizinhanças"],
        ["4. Construtor", "Coordenadas ou embeddings", "Arestas candidatas", "Definir vizinhos locais", "Ficar esparso, denso ou fragmentado"],
        ["5. VGAE", "Grafo candidato", "Arestas refinadas", "Recuperar similaridades úteis", "Adicionar relações espúrias"],
        ["6. Downstream GNN", "Topologia final", "Predições", "Medir utilidade real", "Interagir mal com a nova topologia"],
        ["7. Seleção", "Scores de validação", "Decisão final", "Evitar rewiring prejudicial", "Sobreajuste se reutilizar a mesma validação"],
    ]
    story.append(base.data_table(stages[0], stages[1:], [2.5 * cm, 2.8 * cm, 2.7 * cm, 4.4 * cm, 4.2 * cm], st, font_size=6.9, header_color=base.NAVY))
    story.append(base.heading("23.1 Por que usar um Auxiliary GCN", st["h2"], 1))
    story.append(base.p("As features brutas podem não colocar nós semanticamente parecidos perto uns dos outros. O Auxiliary GCN aprende uma nova representação orientada pela tarefa. A intenção não é usar esse modelo como resposta final, mas usar seu espaço latente para decidir quais nós deveriam ser vizinhos. O controle 'Auxiliary GCN only' foi acrescentado justamente para separar essas duas funções.", st["body"]))
    story.append(base.heading("23.2 Por que reduzir para duas dimensões", st["h2"], 1))
    story.append(base.p("A triangulação de Delaunay é naturalmente definida em um espaço geométrico de baixa dimensão. O UMAP tenta preservar vizinhanças locais do embedding em duas dimensões. Isso simplifica a construção e produz uma malha esparsa. A desvantagem é que qualquer redução pode distorcer distâncias, razão pela qual a trustworthiness foi medida.", st["body"]))
    story.append(base.heading("23.3 O que significa a porcentagem de rewiring", st["h2"], 1))
    story.append(base.p("A porcentagem controla quanto da estrutura candidata entra no grafo final. Valores pequenos fazem uma intervenção conservadora; valores altos substituem mais relações. A escolha deve ocorrer na topology-validation. Não se escolhe primeiro Delaunay usando o teste e depois a melhor porcentagem: todos os candidatos e taxas permitidos devem concorrer somente na validação.", st["body"]))
    story.append(qbox("Quando os percentuais entram?", "Desde a seleção de topologia. Cada combinação de construtor e percentual é um candidato. A validação escolhe a combinação completa; depois o teste é consultado uma única vez.", st))

    story += [PageBreak(), base.heading("24. Como foi garantida uma comparação justa", st["h1"], 0)]
    story.append(base.p("'Usar as mesmas condições' significa que as diferenças observadas devem vir do método, não de vantagens experimentais escondidas. Para isso, o estudo iguala dados, splits, seeds, backbone, critério de seleção, orçamento de ajuste e forma de medir.", st["body"]))
    fairness = [
        ["Elemento controlado", "O que deve ser igual", "Problema evitado"],
        ["Dataset e versão", "Mesmos nós, features, rótulos e arestas de entrada", "Comparar tarefas diferentes"],
        ["Split", "Mesmos nós em treino, validação e teste", "Uma divisão mais fácil favorecer um método"],
        ["Seed", "Mesma aleatoriedade pareada", "Confundir sorte com ganho"],
        ["Backbone", "Mesma arquitetura downstream", "Atribuir ao rewiring um ganho do modelo"],
        ["Early stopping", "Mesmo critério e paciência", "Treinar um método por mais tempo"],
        ["Tuning", "Orçamento comparável", "Dar mais tentativas a um concorrente"],
        ["Seleção", "Somente validação", "Vazamento do teste"],
        ["Métrica", "Mesmo macro-F1 e mesma agregação", "Comparar números incompatíveis"],
    ]
    story.append(base.data_table(fairness[0], fairness[1:], [3.7 * cm, 6.6 * cm, 6.3 * cm], st, font_size=7.6, header_color=base.GREEN))
    story.append(base.heading("24.1 Comparação pareada", st["h2"], 1))
    story.append(base.p("Em cada seed, o baseline e o método usam a mesma divisão dos dados. Calcula-se o ganho dentro da seed. Essa diferença pareada remove parte da variação causada pelo split e pela inicialização. Comparar apenas duas médias independentes desperdiçaria essa informação.", st["body"]))
    story.append(formula_box("Ganho por seed", "ganho_s = F1_selecionado,s - F1_original,s", "Um ganho de 0,054 equivale a +5,4 pontos percentuais. O cálculo deve ser feito antes de resumir as seeds.", st))
    story.append(base.heading("24.2 O protocolo final de validação", st["h2"], 1))
    validation = [
        ["Parte", "Percentual no piloto", "Uso permitido"],
        ["Train", "60%", "Ajustar parâmetros do Auxiliary GCN e das GNNs"],
        ["Inner-validation", "10%", "Early stopping e checkpoint do Auxiliary GCN"],
        ["Topology-validation", "10%", "Escolher original, construtor e percentual"],
        ["Test", "20%", "Avaliar somente a decisão congelada"],
    ]
    story.append(base.data_table(validation[0], validation[1:], [4.0 * cm, 3.1 * cm, 9.5 * cm], st, font_size=8.1, header_color=base.BLUE))
    story.append(base.callout("A separação reduz a chance de escolher uma topologia que apenas se adaptou ao mesmo conjunto usado para parar o encoder.", st["callout"], base.PALE_GREEN, base.GREEN))

    story += [PageBreak(), base.heading("25. Estatística sem mistério", st["h1"], 0)]
    stat_rows = [
        ["Medida", "Pergunta respondida", "Como interpretar"],
        ["Média do ganho", "Qual foi o efeito médio?", "Soma dos ganhos dividida pelo número de execuções"],
        ["Mediana", "Qual ganho representa o centro sem ser dominado por extremos?", "Metade das execuções fica abaixo e metade acima"],
        ["Desvio-padrão", "O resultado varia muito?", "Maior valor significa mais instabilidade entre seeds"],
        ["IC bootstrap", "Qual faixa é compatível com o ganho médio?", "Se todo o intervalo fica acima de zero, a evidência é mais forte"],
        ["Wilcoxon", "As diferenças pareadas tendem a ter o mesmo sinal?", "p pequeno rejeita a hipótese de diferenças centradas em zero"],
        ["Runs positivas", "Com que frequência houve qualquer melhoria?", "Mede consistência, não tamanho"],
        ["Ganho > 0,5 p.p.", "Com que frequência o efeito foi relevante?", "Evita chamar variações minúsculas de sucesso"],
        ["Cohen dz", "Qual o efeito em unidades de variabilidade?", "Útil para comparar força, mas não substitui pontos percentuais"],
    ]
    story.append(base.data_table(stat_rows[0], stat_rows[1:], [3.1 * cm, 6.5 * cm, 7.0 * cm], st, font_size=7.6, header_color=base.NAVY))
    story.append(base.heading("25.1 Por que Wilcoxon", st["h2"], 1))
    story.append(base.p("Antes, a análise podia depender principalmente de médias, intervalos ou teste t. O Wilcoxon pareado foi adotado porque trabalha com a ordem e o sinal das diferenças entre pares e exige menos suposições sobre normalidade. Ele é apropriado quando há várias seeds pareadas e a distribuição dos ganhos pode ser assimétrica ou conter outliers.", st["body"]))
    story.append(qbox("p < 0,05 prova que o método é bom?", "Não sozinho. O p-valor indica incompatibilidade com efeito central zero, mas não informa se o ganho é grande, frequente ou útil. Por isso ele é lido junto com ganho em p.p., intervalo de confiança e fração de runs positivas.", st))
    story.append(base.heading("25.2 Bootstrap", st["h2"], 1))
    story.append(base.p("O bootstrap sorteia repetidamente amostras com reposição dos ganhos observados e recalcula a média. Os percentis dessas médias formam um intervalo de confiança. É uma forma prática de representar incerteza sem depender de uma fórmula normal rígida.", st["body"]))
    story.append(base.heading("25.3 Unidade independente", st["h2"], 1))
    story.append(base.p("No experimento de gate havia 1.080 linhas por seed, mas seis seeds pertenciam ao mesmo grafo. Tratar todas como grafos independentes inflaria a evidência. Por isso o resultado principal usa 180 médias por grafo. Essa decisão gera os números +5,398 p.p., 93,9% positivos e 89,4% acima de +0,5 p.p.", st["body"]))
    story.append(base.heading("25.4 Múltiplas comparações", st["h2"], 1))
    story.append(base.p("Ao testar muitos datasets, backbones e candidatos, algum p-valor pequeno pode aparecer por acaso. A defesa correta não depende de um único p-valor isolado. Ela procura padrões repetidos, tamanhos de efeito, intervalos, validação cega e coerência entre experimentos. Se o paper fizer uma afirmação confirmatória global, deve declarar uma correção para múltiplos testes ou separar análises confirmatórias de exploratórias.", st["body"]))

    story += [PageBreak(), base.heading("26. Dicionário completo das métricas", st["h1"], 0)]
    metric_rows = [
        ["Métrica", "O que mede", "Sinal desejável?", "Cuidado"],
        ["Accuracy", "Proporção total de acertos", "Maior", "Pode esconder classes pequenas"],
        ["Macro-F1", "F1 calculado por classe e depois promediado", "Maior", "Cada classe pesa igualmente"],
        ["Densidade", "Fração das arestas possíveis que existe", "Depende", "Grafos maiores tendem a ser esparsos"],
        ["Grau médio", "Número médio de vizinhos", "Depende", "Mais vizinhos também podem trazer ruído"],
        ["CV do grau", "Desigualdade relativa entre graus", "Depende", "Hubs podem ajudar ou concentrar fluxo"],
        ["Componentes", "Quantidade de partes desconectadas", "Menor em geral", "Comunidades reais podem ser separadas"],
        ["Maior componente", "Fração de nós no maior bloco conectado", "Maior em geral", "Não mede qualidade semântica"],
        ["Isolados", "Nós sem vizinhos", "Menor", "Zero não garante boa topologia"],
        ["Clustering", "Fechamento local de triângulos", "Depende", "Pode refletir redundância local"],
        ["Transitividade", "Triângulos em relação a triplas conectadas", "Depende", "É global e sensível ao grau"],
        ["Assortatividade", "Nós de graus semelhantes se conectam?", "Depende", "Não é homofilia de classe"],
        ["Homofilia", "Arestas entre nós da mesma classe", "Frequentemente maior", "Pode ser enganosa em tarefas heterofílicas"],
        ["lambda2", "Conectividade espectral e dificuldade de separar o grafo", "Intermediário", "Maior não implica melhor classificação"],
        ["Cheeger", "Força aproximada do gargalo de corte", "Intermediário", "Limites, não valor exato do melhor corte"],
        ["Caminho médio", "Distância média entre pares", "Menor em geral", "Atalhos ruins também reduzem"],
        ["Diâmetro", "Maior distância aproximada", "Menor em geral", "Pode cair por arestas espúrias"],
        ["Resistência efetiva", "Redundância de caminhos entre pares", "Menor em geral", "Depende da amostragem de pares"],
        ["Trustworthiness", "Preservação dos vizinhos após UMAP", "Maior", "Não mede utilidade para o rótulo"],
    ]
    story.append(base.data_table(metric_rows[0], metric_rows[1:], [3.0 * cm, 5.1 * cm, 2.6 * cm, 5.9 * cm], st, font_size=6.5, header_color=base.TEAL))
    story.append(base.heading("26.1 lambda2 em linguagem simples", st["h2"], 1))
    story.append(base.p("lambda2 é o segundo menor autovalor da Laplaciana normalizada. Se for muito próximo de zero, o grafo pode estar desconectado ou possuir um gargalo forte. Valores maiores indicam que o grafo é mais difícil de separar em dois blocos com poucas arestas. Isso pode facilitar propagação, mas conectividade excessiva também pode misturar classes.", st["body"]))
    story.append(base.heading("26.2 Cheeger", st["h2"], 1))
    story.append(base.p("A constante de Cheeger descreve quão barato é cortar o grafo em duas partes. Como calculá-la exatamente é difícil, usamos limites relacionados ao espectro. Um valor baixo sugere um gargalo. O resultado real mostrou correlação positiva entre o limite superior de Cheeger e o ganho, mas isso foi observado em apenas 12 datasets e não virou regra preditiva robusta.", st["body"]))
    story.append(base.heading("26.3 Trustworthiness do UMAP", st["h2"], 1))
    story.append(base.p("A trustworthiness pergunta quantos vizinhos que aparecem próximos no mapa 2D já eram vizinhos no espaço original de alta dimensão. Valor alto significa menor invenção local pela projeção. Mesmo assim, uma projeção fiel pode preservar vizinhanças inúteis para a tarefa, e uma projeção menos fiel pode coincidentemente separar melhor as classes. Por isso ela é um diagnóstico, não a métrica final.", st["body"]))

    story += [PageBreak(), base.heading("27. Resultados reais: leitura completa", st["h1"], 0)]
    ds = real.groupby("dataset").agg(
        mean_gain=("gain_f1_mean", "mean"),
        best_gain=("gain_f1_mean", "max"),
        worst_gain=("gain_f1_mean", "min"),
        baseline=("baseline_f1_mean", "mean"),
    ).reset_index().sort_values("mean_gain", ascending=False)
    rows = []
    for r in ds.itertuples(index=False):
        positive = "Favorável" if r.mean_gain > 0.005 else ("Neutro" if r.mean_gain >= -0.005 else "Desfavorável")
        rows.append([r.dataset, f"{100*r.baseline:.2f}%", f"{100*r.mean_gain:+.2f} p.p.", f"{100*r.best_gain:+.2f}", f"{100*r.worst_gain:+.2f}", positive])
    story.append(base.data_table(["Dataset", "Baseline F1", "Ganho médio", "Melhor backbone", "Pior backbone", "Leitura"], rows, [3.3 * cm, 2.6 * cm, 2.7 * cm, 2.7 * cm, 2.7 * cm, 2.6 * cm], st, font_size=7.0, header_color=base.NAVY))
    story.append(base.heading("27.1 Como ler a heterogeneidade", st["h2"], 1))
    story.append(base.p("A média por dataset resume três backbones e pode esconder interações. Cornell, por exemplo, favoreceu GCN e GAT, mas prejudicou GraphSAGE. Minesweeper teve ganho com GCN e perdas fortes com GAT e GraphSAGE. Portanto, a unidade científica importante é dataset mais backbone, não apenas dataset.", st["body"]))
    story.append(base.heading("27.2 Casos fortes", st["h2"], 1))
    best = real.nlargest(8, "gain_f1_mean")[["dataset", "Modelo", "baseline_f1_mean", "selected_f1_mean", "gain_f1_mean", "gain_f1_ci95_lower", "gain_f1_ci95_upper"]]
    best_rows = [[r.dataset, r.Modelo.upper(), f"{100*r.baseline_f1_mean:.2f}", f"{100*r.selected_f1_mean:.2f}", f"{100*r.gain_f1_mean:+.2f}", f"[{100*r.gain_f1_ci95_lower:+.2f}; {100*r.gain_f1_ci95_upper:+.2f}]"] for r in best.itertuples(index=False)]
    story.append(base.data_table(["Dataset", "Backbone", "Original", "Selecionado", "Ganho p.p.", "IC95"], best_rows, [3.7 * cm, 2.2 * cm, 2.5 * cm, 2.7 * cm, 2.3 * cm, 3.2 * cm], st, font_size=7.3, header_color=base.GREEN))
    story.append(base.heading("27.3 Casos negativos", st["h2"], 1))
    worst = real.nsmallest(8, "gain_f1_mean")[["dataset", "Modelo", "gain_f1_mean", "gain_f1_ci95_lower", "gain_f1_ci95_upper"]]
    worst_rows = [[r.dataset, r.Modelo.upper(), f"{100*r.gain_f1_mean:+.2f}", f"[{100*r.gain_f1_ci95_lower:+.2f}; {100*r.gain_f1_ci95_upper:+.2f}]", "Não aplicar sem gate"] for r in worst.itertuples(index=False)]
    story.append(base.data_table(["Dataset", "Backbone", "Ganho p.p.", "IC95", "Decisão"], worst_rows, [4.0 * cm, 2.5 * cm, 2.6 * cm, 3.6 * cm, 3.9 * cm], st, font_size=7.4, header_color=base.RED))
    story.append(base.callout("Resultado importante: os casos negativos não enfraquecem a tese condicional. Eles mostram por que o original precisa participar da seleção e por que a aplicação sem validação seria incorreta.", st["callout"], base.PALE_ORANGE, base.ORANGE))

    story += [base.heading("28. O que as correlações estruturais mostraram", st["h1"], 0)]
    cc = corr[corr["target"].eq("corrected_mean_gain")].copy().sort_values("spearman_rho", ascending=False)
    corr_rows = [[r.metric, f"{r.pearson_r:+.3f}", f"{r.pearson_p:.4f}", f"{r.spearman_rho:+.3f}", f"{r.spearman_p:.4f}", str(int(r.n))] for r in cc.itertuples(index=False)]
    story.append(base.data_table(["Métrica", "Pearson r", "p", "Spearman rho", "p", "n"], corr_rows, [5.0 * cm, 2.5 * cm, 2.0 * cm, 2.7 * cm, 2.0 * cm, 1.4 * cm], st, font_size=7.4, header_color=base.TEAL))
    story.append(base.heading("28.1 Pearson e Spearman", st["h2"], 1))
    story.append(base.p("Pearson mede associação linear: os pontos devem se aproximar de uma reta. Spearman usa a ordem dos valores e detecta relações monotônicas mesmo quando não são lineares. Neste estudo, lambda2, Cheeger e densidade apareceram associados ao ganho, enquanto homofilia de aresta foi fraca para o ganho, apesar de se relacionar com o baseline.", st["body"]))
    story.append(base.heading("28.2 Por que correlação não virou regra", st["h2"], 1))
    story += base.bullets([
        "A amostra tinha somente 12 datasets, então um ou dois casos extremos influenciam muito.",
        "As métricas são correlacionadas entre si; densidade, grau e conectividade não são sinais independentes.",
        "A relação observada em grafos reais pode não se repetir em famílias sintéticas.",
        "A validação cega mostrou baixa capacidade de separar sucesso e fracasso usando apenas a topologia original.",
    ], st["bullet"])
    story.append(qbox("Posso dizer que densidade indica quando usar?", "Pode dizer que densidade foi um indício exploratório nos 12 datasets reais, mas não uma regra validada. A decisão prática deve combinar sinais e, principalmente, uma pequena validação do candidato.", st))

    story += [PageBreak(), base.heading("29. Delaunay e outros construtores", st["h1"], 0)]
    avg_const = constructors.groupby(["method", "backbone"])["gain_f1_mean"].mean().reset_index()
    const_rows = []
    for r in avg_const.sort_values("gain_f1_mean", ascending=False).itertuples(index=False):
        const_rows.append([r.method.replace("_", " "), r.backbone.upper(), f"{100*r.gain_f1_mean:+.3f} p.p."])
    story.append(base.data_table(["Construtor", "Backbone", "Ganho médio nos datasets"], const_rows, [7.2 * cm, 3.0 * cm, 6.4 * cm], st, font_size=7.4, header_color=base.BLUE))
    story.append(base.heading("29.1 O que cada construtor faz", st["h2"], 1))
    methods = [
        ["Método", "Regra", "Vantagem", "Risco"],
        ["Delaunay", "Conecta pontos cujas regiões geométricas são vizinhas", "Malha esparsa sem escolher k", "Depende da projeção 2D"],
        ["kNN", "Cada nó liga aos k mais próximos", "Simples e controlável", "Pode criar ligações assimétricas e hubs"],
        ["Mutual-kNN", "Só mantém vizinhança recíproca", "Mais conservador", "Pode fragmentar"],
        ["Radius", "Liga pares dentro de um raio", "Adapta ao espaço local", "Densidade varia muito"],
        ["MST+kNN", "Garante uma árvore e acrescenta vizinhos", "Evita desconexão", "A árvore pode criar pontes artificiais"],
        ["Random degree-matched", "Aleatório com graus semelhantes", "Controla distribuição de grau", "Sem semântica"],
        ["Random edge-budget", "Aleatório com mesmo número de arestas", "Controla densidade", "Sem estrutura local"],
    ]
    story.append(base.data_table(methods[0], methods[1:], [3.5 * cm, 5.2 * cm, 4.0 * cm, 3.9 * cm], st, font_size=7.3, header_color=base.NAVY))
    story.append(base.callout("Conclusão do controle: Delaunay é uma forma útil de impor localidade, mas não é a única responsável pelo ganho. kNN e mutual-kNN também funcionaram. A representação aprendida é uma parte central da contribuição.", st["callout"], base.PALE_GREEN, base.GREEN))
    story.append(base.heading("29.2 Por que controles aleatórios são necessários", st["h2"], 1))
    story.append(base.p("Se qualquer grafo com o mesmo número de arestas produzisse o mesmo ganho, a geometria não teria valor. O random edge-budget testa densidade; o random degree-matched testa a distribuição de graus. Quando o construtor informado supera esses controles, há evidência de que as relações escolhidas importam, e não apenas a quantidade de arestas.", st["body"]))

    story += [PageBreak(), base.heading("30. Comparação protocol-matched com SDRF e DiffWire", st["h1"], 0)]
    sota_avg = sota.groupby(["method", "backbone"])["gain_f1_mean"].agg(["mean", "median", "count"]).reset_index().sort_values("mean", ascending=False)
    sr = [[r.method, r.backbone.upper(), str(int(r["count"])), f"{100*r['mean']:+.3f}", f"{100*r['median']:+.3f}"] for _, r in sota_avg.iterrows()]
    story.append(base.data_table(["Método", "Backbone", "Combinações", "Ganho médio p.p.", "Mediana p.p."], sr, [4.3 * cm, 2.8 * cm, 3.1 * cm, 3.2 * cm, 3.2 * cm], st, font_size=7.7, header_color=base.NAVY))
    story.append(base.heading("30.1 Por que foi necessário repetir", st["h2"], 1))
    story.append(base.p("Os valores antigos de SDRF e DiffWire vinham de execuções anteriores com protocolos diferentes. Isso impedia afirmar superioridade ou inferioridade: diferenças poderiam vir de splits, seeds, tuning ou seleção. O rerun colocou os métodos sob as mesmas condições do método proposto.", st["body"]))
    story.append(base.heading("30.2 Interpretação", st["h2"], 1))
    story.append(base.p("SDRF ficou próximo de zero em média em muitas combinações, com ganhos e perdas pequenos. DiffWire CT apresentou ganhos em alguns casos, especialmente Roman-Empire com GCN, e perdas em outros. Isso reforça que rewiring não tem um vencedor absoluto e que dataset e backbone determinam a utilidade.", st["body"]))
    story.append(base.heading("30.3 Custo computacional", st["h2"], 1))
    story.append(base.p("O custo é cientificamente relevante porque um ganho pequeno pode não compensar horas ou memória adicionais. Porém, os experimentos não instrumentaram wall-clock, RAM, VRAM e custo de tuning de todos os métodos de forma uniforme. Logo, é correto discutir complexidade operacional e relatar os tempos disponíveis, mas não afirmar uma vantagem quantitativa final de velocidade sem uma medição padronizada.", st["body"]))
    story.append(qbox("Posso dizer que nosso método é mais rápido que SDRF?", "Somente se a versão final do paper incluir medições uniformes. Hoje você pode dizer que há motivação de custo e que o pipeline oferece alternativas simples, como raw kNN, mas a comparação temporal completa ainda não foi fechada.", st))

    story += [PageBreak(), base.heading("31. Por que o primeiro estudo sintético foi negativo", st["h1"], 0)]
    fam_rows = [[r.family.replace("_", " "), str(int(r.graphs)), f"{100*r.mean_geometry_gain:+.2f}", f"{100*r.mean_vgae_increment:+.2f}", f"{100*r.mean_selected_gain:+.2f}", f"{100*r.positive_fraction:.1f}%"] for r in family.sort_values("mean_selected_gain", ascending=False).itertuples(index=False)]
    story.append(base.data_table(["Família", "Grafos", "Geometria p.p.", "Incremento VGAE", "Final p.p.", "Positivos"], fam_rows, [4.7 * cm, 1.7 * cm, 2.7 * cm, 2.8 * cm, 2.4 * cm, 2.3 * cm], st, font_size=7.1, header_color=base.NAVY))
    story.append(base.heading("31.1 A explicação principal", st["h2"], 1))
    story.append(base.p("Nos primeiros sintéticos, o grafo original era frequentemente gerado de forma coerente com a tarefa. Portanto, ele já era uma boa estrutura. Reconstruí-lo a partir de uma projeção 2D introduzia um problema artificial: o método precisava superar um grafo quase ideal. Nos dados reais, o grafo observado pode estar incompleto, ruidoso ou criado por outro processo.", st["body"]))
    story.append(base.heading("31.2 O que aprendemos sobre o VGAE", st["h2"], 1))
    story.append(base.p("A geometria pura foi negativa em todas as famílias em média. O incremento do VGAE foi positivo e recuperou uma parte substancial da perda. Isso apoia a interpretação de que o refinamento por similaridade atua como reparador: ele não cria sozinho todo o ganho, mas recupera relações eliminadas pela regra geométrica.", st["body"]))
    story.append(base.callout("Resultado negativo útil: o primeiro sintético não refutou os resultados reais; mostrou que o cenário de teste favorecia o grafo original e revelou o papel reparador do VGAE.", st["callout"], base.PALE_ORANGE, base.ORANGE))

    story += [PageBreak(), base.heading("32. Corrupção-recuperação e o gate de validação", st["h1"], 0)]
    story.append(base.p("O segundo programa sintético mudou a pergunta. Em vez de gerar um grafo ideal e perguntar se o rewiring o supera, ele gerou uma estrutura latente e corrompeu o grafo observado. Isso aproxima o experimento da hipótese real: o método deve ajudar quando as arestas observadas não representam bem as relações úteis.", st["body"]))
    recovery = [
        ["Condição", "O que representa", "O que se testa"],
        ["Features informativas", "As features preservam a estrutura latente", "Se vizinhos úteis podem ser recuperados"],
        ["Features parciais", "Parte do sinal foi perdida", "Robustez com informação incompleta"],
        ["Features ruidosas", "Sinal fraco ou contaminado", "Limite da reconstrução por similaridade"],
        ["Corrupção de arestas", "Grafo observado difere do latente", "Capacidade de reparar topologia"],
    ]
    story.append(base.data_table(recovery[0], recovery[1:], [4.0 * cm, 6.4 * cm, 6.2 * cm], st, font_size=8.0, header_color=base.TEAL))
    g = gate[gate["aggregation_level"].eq("graph_mean_primary")].iloc[0]
    story.append(base.callout(f"Resultado operacional principal: ganho médio de {g.mean_gain_pp:.3f} p.p.; {100*g.positive_fraction:.1f}% dos grafos positivos; {100*g.gain_gt_0_5pp_fraction:.1f}% acima de +0,5 p.p.; IC95 bootstrap [{g.bootstrap_ci95_low_pp:.3f}; {g.bootstrap_ci95_high_pp:.3f}] p.p.", st["callout"], base.PALE_GREEN, base.GREEN))
    story.append(base.heading("32.1 Como o gate funciona", st["h2"], 1))
    story += base.bullets([
        "Treina-se o baseline no grafo original.",
        "Treinam-se poucos candidatos de rewiring sob o mesmo split.",
        "Compara-se o macro-F1 na topology-validation.",
        "Só se aceita rewiring se o melhor candidato superar o original por pelo menos 1 p.p.",
        "Se a margem não for atingida, mantém-se o original.",
        "A decisão é congelada antes de abrir o test.",
    ], st["bullet"])
    story.append(base.heading("32.2 Por que a margem é útil", st["h2"], 1))
    story.append(base.p("Sem margem, uma diferença minúscula na validação pode ser apenas ruído e provocar uma troca prejudicial. A margem funciona como custo de mudança: o candidato precisa demonstrar vantagem suficiente antes de substituir o original. O valor de 1 p.p. foi eficaz neste estudo, mas deve ser tratado como hiperparâmetro exploratório até uma confirmação externa.", st["body"]))

    story += [PageBreak(), base.heading("33. Teste cego: por que as métricas não bastaram", st["h1"], 0)]
    y = blind["success"].astype(int)
    pred = blind["predicted_success"].astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    acc = (tp + tn) / len(blind)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    blind_rows = [
        ["Casos", str(len(blind))], ["Verdadeiros positivos", str(tp)], ["Verdadeiros negativos", str(tn)],
        ["Falsos positivos", str(fp)], ["Falsos negativos", str(fn)], ["Acurácia", f"{100*acc:.1f}%"],
        ["Precisão para sucesso", f"{100*precision:.1f}%"], ["Recall de sucessos", f"{100*recall:.1f}%"],
    ]
    story.append(base.data_table(["Medida", "Resultado"], blind_rows, [8.3 * cm, 8.3 * cm], st, font_size=8.4, header_color=base.NAVY))
    story.append(base.heading("33.1 O que significa 'cego'", st["h2"], 1))
    story.append(base.p("As previsões foram congeladas antes de revelar o ganho real dos novos grafos. Isso evita ajustar a regra depois de conhecer a resposta. É mais rigoroso que descobrir uma regra e avaliá-la nos mesmos dados usados para criá-la.", st["body"]))
    story.append(base.heading("33.2 Por que a regra falhou", st["h2"], 1))
    story.append(base.p("O sucesso depende de informação ausente nas métricas do grafo original: qualidade das features, rótulos, representação aprendida, distorção do UMAP, construtor, taxa e backbone. Dois grafos com densidade e lambda2 parecidos podem ter relações semânticas completamente diferentes. Assim, a topologia original sozinha tem pouco poder para antecipar o efeito completo do pipeline.", st["body"]))
    story.append(base.heading("33.3 O indicador que ainda pode ser usado", st["h2"], 1))
    story.append(base.p("As métricas podem servir como triagem. Um grafo extremamente fragmentado, com muitos isolados ou forte gargalo, pode justificar testar reconstrução. Mas a decisão final deve ser tomada por validação. O melhor indicador prático encontrado foi o ganho do próprio candidato na topology-validation com uma margem de segurança.", st["body"]))

    story += [PageBreak(), base.heading("34. O teste final: Auxiliary GCN e validações separadas", st["h1"], 0)]
    sel = nested[nested["comparison"].eq("selected_vs_original")].copy().sort_values("mean_gain_pp", ascending=False)
    aux = nested[nested["comparison"].eq("auxiliary_vs_original")][["dataset", "backbone", "mean_gain_pp"]].rename(columns={"mean_gain_pp": "aux_gain"})
    merged = sel.merge(aux, on=["dataset", "backbone"], how="left")
    nr = [[r.dataset, r.backbone.upper(), f"{r.mean_gain_pp:+.2f}", f"[{r.bootstrap_ci95_low_pp:+.2f}; {r.bootstrap_ci95_high_pp:+.2f}]", f"{r.positive_runs_pct:.0f}%", f"{r.wilcoxon_p_value:.4f}", f"{r.aux_gain:+.2f}"] for r in merged.itertuples(index=False)]
    story.append(base.data_table(["Dataset", "Backbone", "Reconstr. vs orig.", "IC95", "Runs +", "Wilcoxon", "Aux vs orig."], nr, [3.1 * cm, 1.8 * cm, 2.8 * cm, 3.2 * cm, 1.8 * cm, 2.0 * cm, 1.9 * cm], st, font_size=6.6, header_color=base.NAVY))
    story.append(base.heading("34.1 O que esse experimento responde", st["h2"], 1))
    story += base.bullets([
        "Compara o Auxiliary GCN como classificador final contra o grafo original.",
        "Compara a GNN downstream no grafo reconstruído contra o original.",
        "Usa inner-validation para checkpoint do encoder.",
        "Usa topology-validation separada para escolher a topologia.",
        "Mantém o teste fora de todas as escolhas.",
        "Executa 4 datasets x 3 backbones x 10 seeds = 120 comparações pareadas principais.",
    ], st["bullet"])
    story.append(base.heading("34.2 A prova mais clara em Roman-Empire", st["h2"], 1))
    story.append(base.p("Em Roman-Empire, o Auxiliary GCN sozinho teve cerca de 2,68% de macro-F1, ou seja, praticamente não resolveu a classificação final. Mesmo assim, a representação produzida por ele permitiu reconstruir uma topologia na qual GAT ganhou +1,83 p.p. em média, com ganho nas 10 seeds e Wilcoxon p = 0,002. GCN ganhou +1,34 p.p., com 70% das seeds positivas e p = 0,049. Isso separa claramente 'o encoder classifica bem' de 'o embedding ajuda a construir um grafo melhor'.", st["body"]))
    story.append(base.callout("Conclusão direta: para o caso positivo mais limpo, o ganho não vem apenas do Auxiliary GCN como classificador. O valor está no uso da representação para reorganizar o fluxo de informação da GNN downstream.", st["callout"], base.PALE_GREEN, base.GREEN))
    story.append(base.heading("34.3 Por que Airports-USA mudou", st["h2"], 1))
    story.append(base.p("O protocolo anterior encontrou ganho grande em Airports-USA. No piloto final, GCN e GraphSAGE ficaram levemente positivos, mas os intervalos cruzaram zero; GAT ficou negativo. Isso pode decorrer de menos seeds, divisão 60/10/10/20, reconstrução por seed e separação da validação. A conclusão correta é que o efeito em Airports-USA é mais sensível ao protocolo do que parecia, enquanto Roman-Empire permaneceu como evidência mais limpa.", st["body"]))

    story += [PageBreak(), base.heading("35. Evolução das hipóteses e decisões", st["h1"], 0)]
    history = [
        ["Fase", "Hipótese ou dúvida", "Teste", "Decisão tomada"],
        ["Inicial", "Delaunay híbrido melhora grafos reais", "12 datasets", "Há ganhos, mas são heterogêneos"],
        ["Estrutural", "Uma métrica explica o ganho", "Correlações e before/after", "Métricas explicam parcialmente"],
        ["Construtores", "O ganho é exclusivo de Delaunay", "kNN, mutual-kNN, radius, MST e aleatórios", "Não; representação e localidade importam"],
        ["SOTA", "Comparação antiga é suficiente", "Rerun SDRF e DiffWire", "Usar somente protocolo compatível"],
        ["Sintético 1", "A mesma vantagem aparecerá em grafos ideais", "Seis famílias", "Não; cenário favorecia o original"],
        ["Sintético 2", "O método recupera grafos corrompidos", "Corrupção-recuperação", "Sim, especialmente com gate"],
        ["Predição", "Topologia original prevê sucesso", "Teste cego", "Não de forma confiável"],
        ["Final", "O encoder sozinho causa o ganho", "Auxiliary only e validações separadas", "Não no caso positivo mais limpo"],
    ]
    story.append(base.data_table(history[0], history[1:], [2.3 * cm, 5.1 * cm, 4.1 * cm, 5.1 * cm], st, font_size=7.0, header_color=base.TEAL))
    story.append(base.heading("35.1 Hipótese final recomendada", st["h2"], 1))
    story.append(base.callout("Uma reconstrução task-informed pode melhorar a classificação quando a representação revela relações discriminativas ausentes ou degradadas no grafo observado, desde que o construtor preserve localidade e conectividade, o refinamento recupere relações úteis e a topologia seja aceita somente após superar o original em validação.", st["callout"], base.PALE_GREEN, base.GREEN))
    story.append(base.heading("35.2 O que não deve ser alegado", st["h2"], 1))
    story += base.bullets([
        "Delaunay é sempre superior.",
        "O método funciona em qualquer grafo.",
        "Densidade ou lambda2 sozinhos determinam sucesso.",
        "O método é definitivamente mais rápido que todo SOTA.",
        "Um p-valor pequeno prova relevância prática.",
        "O teste sintético inicial reproduz exatamente a realidade.",
    ], st["bullet"])
    story.append(base.heading("35.3 O que pode ser alegado com segurança", st["h2"], 1))
    story += base.bullets([
        "Existem combinações dataset-backbone com ganhos grandes, consistentes e estatisticamente sustentados.",
        "A representação aprendida é útil para construir topologia, mesmo quando o classificador auxiliar sozinho é fraco.",
        "Delaunay, kNN e mutual-kNN são alternativas plausíveis; não há construtor universal.",
        "VGAE frequentemente repara perdas introduzidas pela geometria.",
        "Métricas estruturais ajudam na explicação, mas a validação do candidato decide melhor.",
        "Um gate conservador protege contra intervenções prejudiciais.",
    ], st["bullet"])

    story += [PageBreak(), base.heading("36. Roteiro de apresentação", st["h1"], 0)]
    talk = [
        ["Tempo", "Slide", "O que dizer"],
        ["0-1 min", "Problema", "A GNN depende das arestas; o grafo observado pode não ser ideal para a tarefa."],
        ["1-3 min", "Método", "Auxiliary GCN, UMAP, construtor, VGAE e downstream GNN."],
        ["3-4 min", "Protocolo", "Mesmos splits e seeds; seleção por validação; teste fechado."],
        ["4-7 min", "Resultados reais", "Ganhos fortes, perdas e dependência do backbone."],
        ["7-9 min", "Controles", "Outros construtores, aleatórios, SDRF e DiffWire."],
        ["9-11 min", "Sintéticos", "Por que o primeiro foi negativo e o que corrupção-recuperação revelou."],
        ["11-13 min", "Teste final", "Auxiliary only e validações separadas; destaque Roman-Empire."],
        ["13-14 min", "Hipótese final", "Reconstrução útil é condicional e task-informed."],
        ["14-15 min", "Conclusão", "Use gate conservador; métricas explicam, validação decide."],
    ]
    story.append(base.data_table(talk[0], talk[1:], [2.4 * cm, 3.0 * cm, 11.2 * cm], st, font_size=7.8, header_color=base.NAVY))
    story.append(base.heading("36.1 Frases de transição", st["h2"], 1))
    story += base.bullets([
        "Depois do problema: 'A pergunta então deixa de ser qual GNN usar e passa a incluir qual grafo essa GNN deveria receber.'",
        "Antes dos controles: 'Um ganho sozinho não explica a causa; por isso separamos representação, construtor e refinamento.'",
        "Antes dos sintéticos: 'Os dados reais mostram o efeito, mas não controlam o mecanismo; os sintéticos foram usados para isso.'",
        "Antes da conclusão: 'Os resultados negativos foram essenciais porque transformaram uma hipótese ampla em uma regra de uso mais precisa.'",
    ], st["bullet"])
    story.append(base.heading("36.2 Ordem de importância dos números", st["h2"], 1))
    story += base.bullets([
        "Primeiro: +5,398 p.p., 93,9% positivos e 89,4% acima de +0,5 p.p. no gate por grafo.",
        "Segundo: Roman-Empire + GAT com +1,83 p.p., 100% das seeds positivas e p = 0,002 no protocolo final.",
        "Terceiro: Roman-Empire + GCN com +1,34 p.p., 70% positivas e p = 0,049.",
        "Quarto: ganhos reais grandes em Airports e Cornell no protocolo de 100 seeds, acompanhados das perdas em outros casos.",
        "Quinto: Auxiliary GCN de aproximadamente 2,68% em Roman-Empire, mostrando que ele não explica sozinho o ganho downstream.",
    ], st["bullet"])

    story += [PageBreak(), base.heading("37. Perguntas de banca e respostas", st["h1"], 0)]
    qa = [
        ("Qual é a contribuição principal?", "Mostrar que rewiring task-informed pode ser útil de forma condicional, separar os componentes responsáveis e propor uma seleção conservadora por validação."),
        ("Por que Delaunay?", "Ele cria uma malha local esparsa sem exigir k. Porém, os controles mostraram que ele é uma opção, não o único mecanismo válido."),
        ("Por que UMAP em 2D?", "Para tornar possível a construção geométrica e preservar vizinhanças locais. A trustworthiness monitora a distorção."),
        ("Por que não usar diretamente kNN no embedding?", "Isso foi testado. kNN é competitivo e deve ser tratado como candidato, não descartado."),
        ("O Auxiliary GCN não vaza rótulos?", "Ele usa apenas rótulos permitidos pelo treino. Checkpoint e topologia são escolhidos em validações distintas; o teste permanece fechado."),
        ("O ganho não é apenas o encoder?", "Roman-Empire responde isso: o encoder sozinho teve cerca de 2,68% de F1, enquanto a topologia derivada dele melhorou GAT e GCN."),
        ("Por que macro-F1?", "Porque cada classe recebe o mesmo peso, o que é importante em datasets desbalanceados."),
        ("Por que 100 seeds nos reais?", "Para estimar estabilidade, intervalos e testes pareados com menor dependência da sorte."),
        ("Por que 10 seeds no piloto final?", "Foi um teste direcionado a duas dúvidas metodológicas e com maior custo por execução. Ele não substitui os 100 seeds; complementa a evidência."),
        ("Por que Wilcoxon?", "Porque usa diferenças pareadas e depende menos de normalidade do que o teste t."),
        ("Por que também mostrar intervalo bootstrap?", "Porque o p-valor não mostra o tamanho plausível do efeito. O intervalo coloca a incerteza na mesma unidade do ganho."),
        ("Há risco de múltiplas comparações?", "Sim. Por isso a interpretação usa tamanho de efeito, consistência, controles e validação cega, e o paper deve distinguir análises exploratórias das confirmatórias."),
        ("Por que houve perdas?", "Em alguns grafos o original já era adequado; em outros a projeção, o construtor ou o backbone perderam informação útil."),
        ("Resultados negativos invalidam o método?", "Não. Eles definem o domínio de aplicabilidade e justificam manter o original como candidato."),
        ("Por que o sintético inicial foi pior que os reais?", "Porque o grafo sintético original era frequentemente quase ideal. O método tinha pouco a reparar e podia apenas introduzir distorção."),
        ("O segundo sintético foi desenhado depois de ver o primeiro?", "Sim, como investigação mecanística. Isso deve ser declarado; o resultado do gate ainda precisa de confirmação externa para uma alegação confirmatória ampla."),
        ("Densidade prediz sucesso?", "Foi associada ao ganho nos 12 reais, mas falhou como regra geral no teste cego. É indício, não decisão."),
        ("lambda2 alto é sempre melhor?", "Não. Ele indica conectividade espectral, mas conectividade excessiva pode misturar classes."),
        ("Homofilia baixa significa que devo aplicar rewiring?", "Não necessariamente. A utilidade depende do backbone e da informação nas features."),
        ("Qual construtor devo usar?", "Trate original, Delaunay, kNN e mutual-kNN como candidatos; escolha a combinação completa na topology-validation."),
        ("Qual percentual de rewiring usar?", "O percentual é um hiperparâmetro do candidato e deve ser selecionado junto com o construtor, somente na validação."),
        ("Por que um gate de 1 p.p.?", "Ele evita trocar o grafo por diferenças minúsculas de validação. Foi eficaz no estudo, mas deve ser confirmado externamente."),
        ("Por que SDRF e DiffWire foram repetidos?", "Porque resultados antigos com protocolos diferentes não permitem comparação causal justa."),
        ("O método é mais barato?", "Pode oferecer candidatos simples, mas a comparação uniforme de tempo, RAM, VRAM e tuning ainda não foi concluída."),
        ("Qual é o melhor resultado final?", "No protocolo separado, Roman-Empire + GAT: +1,83 p.p., 10 de 10 seeds positivas, p = 0,002."),
        ("Qual é a principal limitação?", "A ausência de um indicador topology-only confiável e a necessidade de validar candidatos por dataset e backbone."),
        ("O que precisa ser feito antes de publicar?", "Alinhar as alegações ao que foi medido, confirmar o gate em uma amostra externa se ele for contribuição central e medir custo apenas se houver alegação de eficiência."),
        ("Qual é a hipótese final em uma frase?", "Rewiring ajuda quando uma representação informativa revela vizinhanças úteis que o grafo observado não expressa bem, e a validação confirma que a intervenção supera o original."),
        ("Isso é uma regra universal?", "Não, e não precisa ser. O resultado científico é uma condição de uso testável e um mecanismo de seleção que evita aplicar o método indiscriminadamente."),
        ("Por que o trabalho é publicável?", "Ele combina resultado empírico, controles causais de componentes, resultados negativos informativos, validação estatística e uma hipótese refinada com limites explícitos."),
    ]
    for i, (q, a) in enumerate(qa, 1):
        story.append(base.heading(f"37.{i} {q}", st["h3"], 1))
        story.append(base.p(a, st["body"]))

    story += [base.heading("38. Folha de consulta rápida para a apresentação", st["h1"], 0)]
    cheat = [
        ["Tema", "Resposta em uma linha"],
        ["Problema", "Arestas definem o fluxo de informação e podem estar desalinhadas com a tarefa."],
        ["Método", "Embedding supervisionado, geometria ou similaridade, refinamento e GNN downstream."],
        ["Métrica principal", "Macro-F1; ganho sempre pareado por seed e expresso em pontos percentuais."],
        ["Seleção", "Checkpoint na inner-validation; grafo na topology-validation; teste fechado."],
        ["Resultado real", "Ganhos fortes em combinações específicas e perdas em outras."],
        ["Delaunay", "Útil, mas não exclusivo; kNN e mutual-kNN também funcionam."],
        ["VGAE", "Principalmente repara relações perdidas pela geometria."],
        ["Indicadores", "Densidade, lambda2 e Cheeger explicam parcialmente, mas não decidem."],
        ["Gate", "+5,398 p.p.; 93,9% positivos; 89,4% acima de +0,5 p.p. por grafo."],
        ["Teste final", "Roman-Empire + GAT: +1,83 p.p.; 100% das seeds; p = 0,002."],
        ["Encoder", "Auxiliary only foi muito fraco em Roman-Empire; o ganho vem do uso topológico da representação."],
        ["Hipótese final", "Reconstrução task-informed ajuda quando repara relações úteis e a validação confirma."],
        ["Limite", "Não há método universal; original sempre deve competir com candidatos."],
    ]
    story.append(base.data_table(cheat[0], cheat[1:], [4.1 * cm, 12.5 * cm], st, font_size=8.0, header_color=base.GREEN))
    story.append(Spacer(1, 0.3 * cm))
    story.append(base.callout("Mensagem final para a banca: o trabalho não tenta provar que Delaunay vence sempre. Ele demonstra quando uma reconstrução informada pela tarefa pode ser útil, quais componentes sustentam o efeito e como impedir que o rewiring seja aplicado quando prejudica.", st["callout"], base.PALE_GREEN, base.GREEN))
    story.append(base.heading("38.1 Checklist antes de apresentar", st["h2"], 1))
    story += base.bullets([
        "Memorize os cinco números prioritários do capítulo 36.",
        "Explique pontos percentuais sem confundir com variação percentual relativa.",
        "Diga espontaneamente que o método é condicional; isso evita uma objeção fácil.",
        "Ao receber uma crítica, identifique se ela trata de efeito, mecanismo, validade ou custo.",
        "Não prometa superioridade temporal sem medição uniforme.",
        "Use os resultados negativos como evidência da evolução metodológica.",
        "Volte sempre à regra: métricas explicam, validação decide, teste confirma.",
    ], st["bullet"])

    return story


def main():
    base.register_fonts()
    base.OUT_DIR.mkdir(parents=True, exist_ok=True)
    st = base.styles()
    old_data = base.load_and_prepare()
    old_charts = base.build_charts(old_data)
    old_story = base.build_story(st, old_data, old_charts)

    first_break = next(i for i, item in enumerate(old_story) if isinstance(item, PageBreak))
    story = cover(st) + old_story[first_break + 1:] + add_extended_manual(st, load_extra())

    doc = base.BookletDoc(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.45 * cm,
        title="Manual detalhado para apresentação e defesa do estudo de rewiring geométrico e híbrido",
        author="Projeto VGAE-Delaunay Hybrid Rewiring",
        subject="Metodologia, métricas, resultados, decisões, conclusões e perguntas de defesa",
    )
    doc.multiBuild(story)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
