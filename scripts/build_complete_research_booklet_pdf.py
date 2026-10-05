#!/usr/bin/env python3
"""Build the complete plain-language research booklet as a polished PDF."""

from __future__ import annotations

from pathlib import Path
from datetime import date
import math

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs" / "apostila_rewiring"
OUT_PDF = OUT_DIR / "Apostila_Completa_Rewiring_Geometrico_Hibrido_20260923.pdf"

REAL = ROOT / "outputs_revision" / "downloaded_results" / "global_12datasets_corrected_20260824"
SOTA = ROOT / "outputs_revision" / "downloaded_results" / "global_protocol_matched_sota_controls_N30_lean"
SYN = ROOT / "outputs_revision" / "downloaded_results" / "synthetic_suitability_pilot_20260917"
REC = ROOT / "outputs_revision" / "downloaded_results" / "corruption_recovery_pilot_20260918"
BLIND = ROOT / "outputs_revision" / "downloaded_results" / "blind_sequential_20260919"
NESTED = ROOT / "outputs_revision" / "downloaded_results" / "nested_validation_auxiliary_pilot_20260920"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2F6B9A")
TEAL = colors.HexColor("#2A7F75")
GREEN = colors.HexColor("#2E7D57")
ORANGE = colors.HexColor("#D17B31")
RED = colors.HexColor("#B84A4A")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GREEN = colors.HexColor("#EAF5F0")
PALE_ORANGE = colors.HexColor("#FBF0E5")
PALE_RED = colors.HexColor("#F9EAEA")
INK = colors.HexColor("#263238")
MUTED = colors.HexColor("#64727D")
GRID = colors.HexColor("#CBD5DC")
WHITE = colors.white


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Italic", "C:/Windows/Fonts/ariali.ttf"))


class BookletDoc(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self._decorate_page))

    def _decorate_page(self, canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(1.7 * cm, A4[1] - 1.35 * cm, A4[0] - 1.7 * cm, A4[1] - 1.35 * cm)
        canvas.setFont("Arial", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.7 * cm, A4[1] - 1.05 * cm, "Rewiring geométrico e híbrido - apostila do estudo")
        canvas.drawRightString(A4[0] - 1.7 * cm, 1.05 * cm, f"Página {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            level = getattr(flowable, "toc_level", None)
            if level is not None:
                text = flowable.getPlainText()
                key = f"heading-{level}-{self.seq.nextf('heading')}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, self.page, key))


def styles():
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Arial",
        fontSize=9.5,
        leading=13.2,
        textColor=INK,
        spaceAfter=7,
    )
    small = ParagraphStyle("Small", parent=body, fontSize=8.1, leading=10.6, textColor=MUTED)
    note = ParagraphStyle("Note", parent=body, fontSize=8.8, leading=12.0, leftIndent=7, rightIndent=7)
    h1 = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName="Arial-Bold",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        spaceBefore=3,
        spaceAfter=9,
        keepWithNext=True,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="Arial-Bold",
        fontSize=13,
        leading=16,
        textColor=BLUE,
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=True,
    )
    h3 = ParagraphStyle(
        "H3",
        parent=base["Heading3"],
        fontName="Arial-Bold",
        fontSize=10.5,
        leading=13,
        textColor=TEAL,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=13, firstLineIndent=-7, bulletIndent=5, spaceAfter=3)
    table = ParagraphStyle("Table", parent=body, fontSize=7.6, leading=9.3, spaceAfter=0)
    table_head = ParagraphStyle("TableHead", parent=table, fontName="Arial-Bold", textColor=WHITE)
    callout = ParagraphStyle("Callout", parent=body, fontName="Arial-Bold", fontSize=10, leading=14, textColor=NAVY)
    cover_title = ParagraphStyle("CoverTitle", parent=h1, fontSize=25, leading=30, textColor=NAVY, alignment=TA_LEFT)
    cover_sub = ParagraphStyle("CoverSub", parent=body, fontSize=13, leading=18, textColor=BLUE)
    return {
        "body": body,
        "small": small,
        "note": note,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "bullet": bullet,
        "table": table,
        "table_head": table_head,
        "callout": callout,
        "cover_title": cover_title,
        "cover_sub": cover_sub,
    }


def heading(text: str, style, level: int):
    paragraph = Paragraph(text, style)
    paragraph.toc_level = level
    return paragraph


def p(text: str, st):
    return Paragraph(text, st)


def bullets(items: list[str], st):
    return [Paragraph(f"• {item}", st) for item in items]


def callout(text: str, st, background=PALE_BLUE, border=BLUE):
    table = Table([[Paragraph(text, st)]], colWidths=[16.6 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 1, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def data_table(headers, rows, widths, st, font_size=7.5, header_color=NAVY, alignments=None):
    cooked = [[Paragraph(str(h), st["table_head"]) for h in headers]]
    for row in rows:
        cooked.append([Paragraph(str(value), st["table"]) for value in row])
    table = Table(cooked, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F5F8FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if alignments:
        for col, alignment in enumerate(alignments):
            commands.append(("ALIGN", (col, 1), (col, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


def fmt_pp(value: float) -> str:
    return f"{value:+.2f} p.p.".replace(".", ",")


def fmt_pct(value: float, digits=1) -> str:
    return f"{value:.{digits}f}%".replace(".", ",")


def save_bar_chart(labels, values, title, subtitle, filename, *, colors_by_sign=True, x_label="Ganho de macro-F1 (p.p.)"):
    """Return a vector horizontal bar chart; filename is kept for call compatibility."""
    width, height = 475, max(245, 72 + 18 * len(labels))
    drawing = Drawing(width, height)
    drawing.add(String(8, height - 18, title, fontName="Arial-Bold", fontSize=12, fillColor=NAVY))
    drawing.add(String(8, height - 33, subtitle, fontName="Arial", fontSize=7.5, fillColor=MUTED))
    left, right, bottom, top = 145, width - 42, 28, height - 53
    minimum = min(min(values), 0.0)
    maximum = max(max(values), 0.0)
    span = maximum - minimum or 1.0
    zero_x = left + (0.0 - minimum) / span * (right - left)
    drawing.add(Line(zero_x, bottom, zero_x, top, strokeColor=INK, strokeWidth=0.7))
    for tick in np.linspace(minimum, maximum, 5):
        x = left + (tick - minimum) / span * (right - left)
        drawing.add(Line(x, bottom, x, top, strokeColor=colors.HexColor("#E5EAEE"), strokeWidth=0.4))
        drawing.add(String(x, 12, f"{tick:.1f}", fontName="Arial", fontSize=6.5, textAnchor="middle", fillColor=MUTED))
    row_h = (top - bottom) / max(len(labels), 1)
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top - (i + 0.5) * row_h
        value_x = left + (value - minimum) / span * (right - left)
        x = min(zero_x, value_x)
        w = max(abs(value_x - zero_x), 0.8)
        fill = GREEN if value >= 0 else RED
        if not colors_by_sign:
            fill = BLUE
        drawing.add(Rect(x, y - min(5.3, row_h * 0.32), w, min(10.6, row_h * 0.64), fillColor=fill, strokeColor=None))
        short_label = label if len(label) <= 27 else label[:26] + "…"
        drawing.add(String(left - 5, y - 2.3, short_label, fontName="Arial", fontSize=6.7, textAnchor="end", fillColor=INK))
        anchor = "start" if value >= 0 else "end"
        offset = 3 if value >= 0 else -3
        drawing.add(String(value_x + offset, y - 2.3, f"{value:+.2f}", fontName="Arial-Bold", fontSize=6.5, textAnchor=anchor, fillColor=INK))
    drawing.add(String((left + right) / 2, 0, x_label, fontName="Arial", fontSize=7, textAnchor="middle", fillColor=MUTED))
    return drawing


def save_grouped_chart(frame, index, column, value, title, filename):
    ordered = frame.sort_values([index, column])
    labels = [f"{getattr(row, index)} / {getattr(row, column).upper()}" for row in ordered.itertuples(index=False)]
    values = ordered[value].tolist()
    return save_bar_chart(labels, values, title, "Cada barra representa um dataset e backbone", filename)


def load_and_prepare():
    real = pd.read_csv(REAL / "final_selected_test_results.csv")
    real["gain_pp"] = 100 * real["gain_f1_mean"]
    real_dataset = real.groupby("dataset", as_index=False)["gain_pp"].mean().sort_values("gain_pp")
    real_backbone = real.groupby("Modelo", as_index=False)["gain_pp"].mean().sort_values("gain_pp")

    constructors = pd.read_csv(SOTA / "constructor_summary.csv")
    constructors["gain_pp"] = 100 * constructors["gain_f1_mean"]
    keep = ["knn", "mutual_knn", "random_degree_matched", "random_edge_budget_matched"]
    constructors_avg = constructors[constructors["method"].isin(keep)].groupby(["method", "backbone"], as_index=False)["gain_pp"].mean()

    sota = pd.read_csv(SOTA / "sota_summary.csv")
    sota["gain_pp"] = 100 * sota["gain_f1_mean"]
    sota_avg = sota.groupby(["method", "backbone"], as_index=False)["gain_pp"].mean()

    nested_selected = pd.read_csv(NESTED / "nested_auxiliary_selected_per_seed.csv")
    nested_summary = pd.read_csv(NESTED / "nested_auxiliary_summary.csv")
    nested_gain = nested_summary[nested_summary["comparison"].eq("selected_vs_original")].copy()

    synthetic_family = pd.read_csv(SYN / "analysis_family_summary.csv")
    gate = pd.read_csv(REC / "validation_gate_1pp_reproduction_summary.csv")
    return real, real_dataset, real_backbone, constructors_avg, sota_avg, nested_selected, nested_summary, nested_gain, synthetic_family, gate


def build_charts(data):
    real, real_dataset, _, constructors_avg, sota_avg, _, _, nested_gain, synthetic_family, _ = data
    charts = {}
    charts["real"] = save_bar_chart(
        real_dataset["dataset"].tolist(),
        real_dataset["gain_pp"].tolist(),
        "Resultados nos 12 datasets reais",
        "Média do ganho entre GCN, GAT e GraphSAGE; seleção por validação",
        "01_real_dataset_gain.png",
    )
    labels = [f"{row.method} / {row.backbone}" for row in constructors_avg.itertuples()]
    charts["constructors"] = save_bar_chart(
        labels,
        constructors_avg["gain_pp"].tolist(),
        "Controles de construtor",
        "Mesma representação aprendida e orçamento comparável de arestas",
        "02_constructor_controls.png",
    )
    sota_labels = [f"{row.method} / {row.backbone}" for row in sota_avg.itertuples()]
    charts["sota"] = save_bar_chart(
        sota_labels,
        sota_avg["gain_pp"].tolist(),
        "Métodos SOTA sob protocolo compatível",
        "Média sobre os 12 datasets; 30 seeds por comparação",
        "03_sota.png",
    )
    syn = synthetic_family.sort_values("mean_selected_gain")
    charts["synthetic"] = save_bar_chart(
        syn["family"].str.replace("_", " ").tolist(),
        (100 * syn["mean_selected_gain"]).tolist(),
        "Primeiro estudo sintético por família",
        "Ganho do pipeline selecionado sobre o grafo original",
        "04_synthetic_family.png",
    )
    nested_plot = nested_gain[["dataset", "backbone", "mean_gain_pp"]]
    charts["nested"] = save_grouped_chart(
        nested_plot,
        "dataset",
        "backbone",
        "mean_gain_pp",
        "Validação separada: ganho da reconstrução sobre o original",
        "05_nested_gain.png",
    )
    recovery_labels = ["Pipeline aprendido", "Delaunay puro", "Raw kNN", "Raw UMAP-Delaunay", "Gate 1 p.p."]
    recovery_values = [-0.802, -2.552, 6.910, 4.420, 5.398]
    charts["recovery"] = save_bar_chart(
        recovery_labels,
        recovery_values,
        "Corrupção-recuperação: representação e seleção importam",
        "Ganho médio sobre o grafo observado",
        "06_recovery.png",
    )
    return charts


def flow_table(st):
    boxes = [
        ("1. Grafo observado", "Features, rótulos de treino e arestas originais"),
        ("2. Auxiliary GCN", "Aprende uma representação supervisionada dos nós"),
        ("3. UMAP em 2D", "Transforma a representação em coordenadas geométricas"),
        ("4. Construtor", "Delaunay, kNN, mutual-kNN ou outro candidato"),
        ("5. Refinamento", "VGAE troca parte das arestas usando similaridade"),
        ("6. Downstream GNN", "GCN, GAT ou GraphSAGE avalia o grafo resultante"),
    ]
    rows = []
    for i, (title, detail) in enumerate(boxes):
        rows.append([Paragraph(f"<b>{title}</b><br/>{detail}", st["note"])])
        if i < len(boxes) - 1:
            rows.append([Paragraph("↓", ParagraphStyle("Arrow", parent=st["body"], alignment=TA_CENTER, fontSize=14, textColor=BLUE))])
    table = Table(rows, colWidths=[15.5 * cm], hAlign="CENTER")
    style = [("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for row in range(0, len(rows), 2):
        style += [
            ("BACKGROUND", (0, row), (0, row), PALE_BLUE if row % 4 == 0 else PALE_GREEN),
            ("BOX", (0, row), (0, row), 0.8, BLUE if row % 4 == 0 else TEAL),
            ("TOPPADDING", (0, row), (0, row), 7),
            ("BOTTOMPADDING", (0, row), (0, row), 7),
        ]
    table.setStyle(TableStyle(style))
    return table


def build_story(st, data, charts):
    real, real_dataset, real_backbone, constructors_avg, sota_avg, nested_selected, nested_summary, nested_gain, synthetic_family, gate = data
    story = []

    # Cover
    story += [Spacer(1, 2.0 * cm), HRFlowable(width="100%", thickness=5, color=TEAL), Spacer(1, 0.8 * cm)]
    story.append(p("Apostila completa do estudo de rewiring geométrico e híbrido", st["cover_title"]))
    story.append(p("Do primeiro protocolo em grafos reais aos controles, estudos sintéticos, validação cega e teste final com validações separadas", st["cover_sub"]))
    story += [Spacer(1, 1.1 * cm)]
    story.append(callout(
        "Ideia central: reconstruir vizinhanças usando uma representação informativa, uma regra geométrica ou de similaridade e uma seleção cuidadosa por validação.",
        st["callout"], PALE_GREEN, TEAL,
    ))
    story += [Spacer(1, 1.2 * cm)]
    cover_rows = [
        ["Escopo", "12 datasets reais, três backbones, controles SOTA, construtores alternativos e dois programas sintéticos"],
        ["Atualização", "23 de setembro de 2026"],
        ["Público", "Leitor que precisa entender o estudo sem depender de conhecimento avançado em teoria de grafos"],
        ["Métrica principal", "Macro-F1 e ganho pareado em pontos percentuais"],
    ]
    story.append(data_table(["Item", "Descrição"], cover_rows, [3.1 * cm, 13.5 * cm], st, font_size=8.4, header_color=NAVY))
    story += [Spacer(1, 2.2 * cm), p("Documento técnico-didático. Todos os números apresentados foram recalculados ou conferidos nos CSVs consolidados do projeto.", st["small"]), PageBreak()]

    # TOC
    story.append(heading("Sumário", st["h1"], 0))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName="Arial-Bold", fontSize=10, leading=15, leftIndent=0, textColor=NAVY),
        ParagraphStyle("TOC2", fontName="Arial", fontSize=9, leading=13, leftIndent=14, textColor=INK),
    ]
    story.append(toc)
    story.append(PageBreak())

    # 1
    story.append(heading("1. O problema estudado", st["h1"], 0))
    story.append(p("Uma rede neural de grafos aprende combinando as features de cada nó com informações recebidas de seus vizinhos. Isso significa que as arestas não são apenas um desenho: elas definem quem troca informação com quem. Se o grafo observado estiver incompleto, ruidoso ou ligado por relações pouco úteis para a tarefa, a GNN pode aprender pior.", st["body"]))
    story.append(p("Rewiring é o processo de modificar essas conexões. O objetivo não é embelezar o grafo nem tornar todas as métricas estruturais maiores. O objetivo é criar vizinhanças que ajudem a classificação sem destruir conectividade importante.", st["body"]))
    story.append(callout("Pergunta central do estudo: quando uma representação aprendida, combinada com geometria e similaridade, consegue criar um grafo mais útil para uma GNN downstream?", st["callout"]))
    story.append(heading("1.1 A hipótese que realmente foi testada", st["h2"], 1))
    story.append(p("A hipótese nunca precisou afirmar que Delaunay seria o melhor construtor em todos os casos. A formulação correta é condicional: uma reconstrução baseada em uma representação com sinal da tarefa, seguida de um construtor geométrico ou de similaridade, pode produzir rewiring útil para certos grafos e backbones.", st["body"]))
    story += bullets([
        "O método pode ajudar quando a topologia original não expressa bem as relações discriminativas.",
        "O método pode piorar quando substitui um grafo que já contém conectividade útil.",
        "O efeito depende do dataset, da representação, do construtor, da intensidade do rewiring e do backbone downstream.",
        "A comparação correta é pareada: mesma seed, mesmo split e diferença entre o resultado reconstruído e o original.",
    ], st["bullet"])

    story.append(heading("1.2 Conceitos essenciais", st["h2"], 1))
    concepts = [
        ["Nó", "Objeto classificado: artigo, aeroporto, usuário, produto ou outro item."],
        ["Aresta", "Relação que permite troca de informação entre dois nós."],
        ["Feature", "Descrição numérica do nó antes do treinamento."],
        ["Embedding", "Representação aprendida que resume informação útil do nó e de sua vizinhança."],
        ["Backbone", "Arquitetura usada para classificação final: GCN, GAT ou GraphSAGE."],
        ["Rewiring", "Remoção, inclusão ou substituição de arestas."],
        ["Macro-F1", "Média do F1 das classes. Dá peso semelhante a classes grandes e pequenas."],
        ["Ponto percentual", "Diferença direta entre percentuais. De 70% para 73% são +3 p.p."],
    ]
    story.append(data_table(["Termo", "Explicação simples"], concepts, [3.2 * cm, 13.4 * cm], st, font_size=8.3))
    story.append(PageBreak())

    # 2 pipeline
    story.append(heading("2. Como funciona o pipeline proposto", st["h1"], 0))
    story.append(flow_table(st))
    story.append(heading("2.1 Auxiliary GCN", st["h2"], 1))
    story.append(p("O Auxiliary GCN é um encoder supervisionado pequeno. Ele vê as features, o grafo original e os rótulos disponíveis no treino. Sua primeira camada gera learned features: coordenadas internas que tendem a aproximar nós úteis para a classificação.", st["body"]))
    story.append(p("Por que ele existe? Features brutas nem sempre formam uma geometria adequada. O encoder tenta transformar os dados em uma representação mais alinhada com a tarefa. O risco é ele também absorver defeitos do grafo original; esse risco apareceu no experimento de corrupção-recuperação.", st["body"]))
    story.append(heading("2.2 UMAP", st["h2"], 1))
    story.append(p("UMAP reduz a representação aprendida para duas dimensões. Isso permite usar triangulação de Delaunay. A redução é uma compressão: preserva parte das vizinhanças locais, mas pode perder informação. Por isso foi medida a trustworthiness do UMAP.", st["body"]))
    story.append(heading("2.3 Delaunay e outros construtores", st["h2"], 1))
    story += bullets([
        "Delaunay conecta pontos de forma geométrica, criando uma malha local sem exigir a escolha direta de k.",
        "kNN conecta cada nó aos k vizinhos mais próximos.",
        "mutual-kNN mantém somente relações em que os dois nós se reconhecem como vizinhos.",
        "radius graph conecta pares abaixo de uma distância definida.",
        "MST+kNN preserva conectividade global com uma árvore geradora mínima e adiciona vizinhanças locais.",
        "Controles aleatórios mantêm número de arestas ou graus, mas retiram o conteúdo semântico das conexões.",
    ], st["bullet"])
    story.append(heading("2.4 Refinamento VGAE", st["h2"], 1))
    story.append(p("O VGAE aprende similaridade estrutural no grafo. No pipeline, ele remove uma fração das arestas menos compatíveis e adiciona relações com maior similaridade. Foram testadas taxas de 0%, 10%, 25%, 40% e 55%. A taxa zero representa a geometria sem refinamento.", st["body"]))
    story.append(heading("2.5 GNN downstream", st["h2"], 1))
    story.append(p("O grafo reconstruído não é avaliado pelo Auxiliary GCN. Ele é entregue a uma GNN final. GCN faz agregação normalizada, GAT aprende pesos de atenção e GraphSAGE agrega informações amostradas dos vizinhos. A interação entre o grafo e esse backbone foi uma das descobertas mais importantes.", st["body"]))

    # 3 methodology
    story.append(PageBreak())
    story.append(heading("3. Metodologia experimental e decisões de justiça", st["h1"], 0))
    story.append(heading("3.1 Comparação pareada", st["h2"], 1))
    story.append(p("Original e reconstruído usam a mesma seed e o mesmo split. Assim, a diferença não é causada por um conjunto de teste mais fácil. O ganho de cada execução é calculado como macro-F1 reconstruído menos macro-F1 original.", st["body"]))
    story.append(heading("3.2 Seleção por validação", st["h2"], 1))
    story.append(p("As taxas e topologias candidatas são comparadas na validação. Somente depois da escolha o desempenho correspondente no teste é consultado. Escolher pelo teste inflaria o resultado; o protocolo foi corrigido para impedir isso.", st["body"]))
    story.append(heading("3.3 Seeds, splits e unidade científica", st["h2"], 1))
    story.append(p("Seeds repetidas medem variação de inicialização e divisão dos dados. Elas são pareadas nas comparações. Em estudos sintéticos, várias seeds do mesmo grafo não foram tratadas como dezenas de grafos independentes; o grafo foi a unidade primária para intervalos e testes.", st["body"]))
    story.append(heading("3.4 Protocolo mais recente com validações separadas", st["h2"], 1))
    split_rows = [
        ["Treino", "60%", "Ajustar os pesos dos modelos"],
        ["Inner-validation", "10%", "Escolher checkpoints do Auxiliary GCN e das GNNs downstream"],
        ["Topology-validation", "10%", "Escolher a topologia e a taxa de rewiring"],
        ["Teste", "20%", "Avaliação final após todas as escolhas"],
    ]
    story.append(data_table(["Parte", "Fração", "Uso"], split_rows, [3.4 * cm, 2.1 * cm, 11.1 * cm], st, font_size=8.4, header_color=TEAL))
    story.append(p("Essa alteração respondeu diretamente ao risco de usar a mesma validação para parar o Auxiliary GCN e escolher a topologia. O piloto final usou quatro datasets representativos, três backbones e 10 seeds: 120 comparações pareadas.", st["body"]))
    story.append(heading("3.5 Testes estatísticos", st["h2"], 1))
    stats_rows = [
        ["Média do ganho", "Tamanho médio do efeito em pontos percentuais."],
        ["Mediana", "Resultado típico com menor influência de casos extremos."],
        ["Desvio-padrão", "Quanto o ganho varia entre seeds."],
        ["Bootstrap IC95%", "Faixa de valores plausíveis obtida por reamostragem."],
        ["Wilcoxon pareado", "Verifica se as diferenças pareadas tendem a ficar de um lado de zero sem exigir normalidade."],
        ["Fração positiva", "Percentual de seeds ou grafos em que houve melhoria."],
        ["Correção de Holm", "Reduz falsos positivos quando muitas hipóteses são testadas."],
    ]
    story.append(data_table(["Métrica ou teste", "O que responde"], stats_rows, [4.0 * cm, 12.6 * cm], st, font_size=8.2))

    # 4 metrics
    story.append(PageBreak())
    story.append(heading("4. Métricas estruturais explicadas", st["h1"], 0))
    metric_rows = [
        ["Densidade", "Proporção das arestas possíveis que realmente existem.", "Alta densidade aumenta caminhos alternativos, mas não garante arestas úteis."],
        ["Grau médio", "Número médio de vizinhos por nó.", "Mostra o orçamento de comunicação disponível."],
        ["Degree CV", "Variação do grau dividida pela média.", "Valores altos indicam hubs e grande desigualdade de conectividade."],
        ["Homofilia", "Fração de arestas que ligam nós da mesma classe.", "Ajuda GNNs homofílicas, mas pode ser enganosa quando classes ou features têm outra organização."],
        ["lambda2", "Segundo autovalor do Laplaciano normalizado.", "Indica conectividade global e dificuldade de separar o grafo em blocos."],
        ["Cheeger", "Limites espectrais relacionados ao menor gargalo.", "Ajuda a descrever cortes e estrangulamentos do grafo."],
        ["Caminho médio", "Número médio de saltos entre pares conectados.", "Caminhos longos dificultam propagação rápida de informação."],
        ["Diâmetro", "Maior distância aproximada dentro da componente principal.", "Mostra a extensão global do grafo."],
        ["Clustering", "Tendência de vizinhos formarem triângulos.", "Mede localidade, mas clustering alto pode coexistir com fragmentação."],
        ["Modularidade", "Força da divisão do grafo em comunidades.", "Comunidades podem ajudar ou atrapalhar, dependendo do alinhamento com as classes."],
        ["Resistência efetiva", "Medida elétrica de redundância dos caminhos.", "Valores altos sugerem menos rotas alternativas."],
        ["Trustworthiness", "Preservação de vizinhos ao projetar para 2D.", "Verifica se o UMAP inventou muitas vizinhanças falsas."],
    ]
    story.append(data_table(["Métrica", "Definição simples", "Por que interessa"], metric_rows, [3.0 * cm, 6.1 * cm, 7.5 * cm], st, font_size=7.5))
    story.append(callout("Nenhuma dessas métricas, sozinha ou em combinação fixa, virou uma regra confiável de aplicação. Elas continuam úteis para explicar o que o rewiring fez e para levantar hipóteses.", st["callout"], PALE_ORANGE, ORANGE))

    # 5 timeline
    story.append(PageBreak())
    story.append(heading("5. Evolução do estudo e das decisões", st["h1"], 0))
    timeline = [
        ["1. Bases reais", "Confirmar se havia ganho em dados reais", "Ganhos fortes em alguns datasets e perdas em outros", "Tratar o método como condicional"],
        ["2. Correção do protocolo", "Eliminar seleção indevida pelo teste", "Taxa escolhida por validação", "Manter teste para avaliação final"],
        ["3. Estrutura", "Buscar relação com lambda2, densidade, Cheeger e homofilia", "Correlações fortes em 12 datasets", "Criar testes sintéticos antes de virar regra"],
        ["4. Controles", "Separar representação, construtor e aleatoriedade", "kNN e mutual-kNN competitivos; aleatórios ruins", "Abandonar a tese de Delaunay único"],
        ["5. SOTA", "Comparar SDRF e DiffWire nas mesmas condições", "Ganhos médios pequenos e heterogêneos", "Incluir custo e protocolo na comparação"],
        ["6. Sintéticos", "Testar causalidade e indicador estrutural", "Pipeline negativo; VGAE reparou Delaunay", "Focar preservação de conectividade"],
        ["7. Corrupção-recuperação", "Criar topologia observada com defeito recuperável", "Features brutas e gate foram fortes", "Elevar representação e validação ao centro da tese"],
        ["8. Validação cega", "Predizer sucesso antes de conhecer o gabarito", "Regras topológicas próximas do acaso", "Não publicar threshold estrutural como regra"],
        ["9. Teste final", "Isolar Auxiliary GCN e separar validações", "Ganho persistiu em Roman-Empire com GAT e GCN", "Confirmar que o efeito não é só o classificador auxiliar"],
    ]
    story.append(data_table(["Etapa", "Pergunta", "Resultado", "Decisão"], timeline, [2.4 * cm, 4.2 * cm, 5.1 * cm, 4.9 * cm], st, font_size=7.2))

    # 6 real data
    story.append(PageBreak())
    story.append(heading("6. Resultados nos 12 datasets reais", st["h1"], 0))
    story.append(p("O protocolo corrigido avaliou Actor, Airports-Brazil, Airports-Europe, Airports-USA, Amazon-Photo, Coauthor-CS, Cora, Cornell, Minesweeper, Pubmed, Roman-Empire e Texas. Foram três backbones e 100 seeds por combinação final.", st["body"]))
    story.append(charts["real"])
    backbone_rows = [[row.Modelo.upper(), fmt_pp(row.gain_pp)] for row in real_backbone.itertuples()]
    story.append(data_table(["Backbone", "Ganho médio nos 12 datasets"], backbone_rows, [7.3 * cm, 9.3 * cm], st, font_size=8.4, header_color=TEAL))
    story.append(p("A média das 36 combinações dataset-backbone foi +2,576 p.p. Foram 22 combinações positivas e pelo menos um backbone positivo em 11 dos 12 datasets. O resultado relevante não é uma média universal; é a existência de regiões de ganho grandes, estáveis e dependentes do backbone.", st["body"]))
    real_rows = []
    for row in real_dataset.sort_values("gain_pp", ascending=False).itertuples():
        positives = int((real[(real["dataset"] == row.dataset)]["gain_pp"] > 0).sum())
        real_rows.append([row.dataset, fmt_pp(row.gain_pp), f"{positives}/3"])
    story.append(data_table(["Dataset", "Ganho médio", "Backbones positivos"], real_rows, [7.1 * cm, 4.7 * cm, 4.8 * cm], st, font_size=8.0))
    story.append(heading("6.1 Leitura principal", st["h2"], 1))
    story += bullets([
        "Airports-Brazil, Airports-USA e Airports-Europe tiveram ganhos grandes no protocolo original corrigido.",
        "Texas, Cornell e Actor apresentaram ganhos menores ou dependentes do backbone.",
        "Amazon-Photo perdeu nos três backbones; Minesweeper e Roman-Empire mostraram forte interação com a arquitetura.",
        "A escolha do backbone não é detalhe: GCN e GAT responderam melhor que GraphSAGE em média.",
    ], st["bullet"])

    # 7 structure
    story.append(PageBreak())
    story.append(heading("7. O que as métricas estruturais mostraram", st["h1"], 0))
    corr_rows = [
        ["lambda2 normalizado", "+0,867", "Grafos mais conectados globalmente pareciam responder melhor"],
        ["Cheeger superior", "+0,867", "Mesma direção de lambda2"],
        ["Caminho médio", "-0,860", "Caminhos menores acompanharam ganhos maiores"],
        ["Diâmetro", "-0,831", "Grafos menos espalhados pareceram mais favoráveis"],
        ["Densidade", "+0,797", "Maior orçamento de arestas acompanhou melhor resposta"],
    ]
    story.append(data_table(["Métrica original", "Spearman com ganho", "Leitura inicial"], corr_rows, [4.3 * cm, 3.4 * cm, 8.9 * cm], st, font_size=8.1))
    story.append(p("Essas correlações eram fortes, mas tinham somente 12 pontos e várias métricas mediam aspectos semelhantes. Os estudos sintéticos e cegos mostraram que elas separavam famílias de grafos, porém não prediziam bem variações dentro da mesma família.", st["body"]))
    story.append(heading("7.1 Before e after", st["h2"], 1))
    story.append(p("O Delaunay tendeu a tornar os grafos mais locais e agrupados. Ao mesmo tempo, reduziu conectividade global, aumentou caminhos e criou componentes adicionais. O VGAE recuperou parte dessas perdas. Isso explica por que aumentar clustering não significou automaticamente melhorar a classificação.", st["body"]))
    story.append(callout("Conclusão estrutural: um bom rewiring precisa preservar conectividade suficiente e, ao mesmo tempo, trocar arestas por relações mais úteis para a tarefa. Melhorar uma única métrica não basta.", st["callout"]))

    # 8 constructors
    story.append(PageBreak())
    story.append(heading("8. Delaunay versus outros construtores", st["h1"], 0))
    story.append(charts["constructors"])
    story.append(p("kNN e mutual-kNN foram tão competitivos quanto o Delaunay em vários cenários. Os controles aleatórios foram muito inferiores. Isso fornece duas respostas importantes.", st["body"]))
    story += bullets([
        "A representação aprendida contém informação útil, pois conexões baseadas nela superam conexões aleatórias com orçamento semelhante.",
        "O ganho não pertence exclusivamente à triangulação de Delaunay. O construtor ideal depende do grafo e do backbone.",
        "A contribuição científica deve ser apresentada como reconstrução task-informed com construtor selecionável, não como superioridade universal de Delaunay.",
    ], st["bullet"])
    story.append(heading("8.1 Por que manter Delaunay na pesquisa", st["h2"], 1))
    story.append(p("Delaunay continua relevante porque oferece uma regra geométrica simples, local e sem um k fixo explícito. Além disso, raw-feature UMAP seguido de Delaunay foi forte no experimento de corrupção. O controle apenas corrige o alcance da tese: Delaunay é uma opção útil, não a única opção.", st["body"]))

    # 9 SOTA
    story.append(PageBreak())
    story.append(heading("9. Comparação com SDRF e DiffWire CT", st["h1"], 0))
    story.append(p("Os valores antigos desses métodos vinham de execuções não totalmente compatíveis. Por isso SDRF e DiffWire CT foram reexecutados com os mesmos datasets, splits, seeds, seleção por validação, orçamento de tuning e avaliação final.", st["body"]))
    story.append(charts["sota"])
    sota_rows = []
    for row in sota_avg.sort_values(["method", "backbone"]).itertuples():
        sota_rows.append([row.method, row.backbone.upper(), fmt_pp(row.gain_pp)])
    story.append(data_table(["Método", "Backbone", "Ganho médio"], sota_rows, [6.0 * cm, 4.2 * cm, 6.4 * cm], st, font_size=8.3))
    story.append(p("DiffWire CT ficou levemente positivo no GCN. SDRF permaneceu próximo de zero ou negativo em média. O método proposto e os melhores construtores simples mostraram ganhos maiores em regimes específicos, mas a comparação deve considerar tempo, memória e o número de candidatos avaliados.", st["body"]))

    # 10 synthetic
    story.append(PageBreak())
    story.append(heading("10. Primeiro estudo sintético", st["h1"], 0))
    story.append(p("Foram criados 205 grafos de seis famílias, 41 configurações, 1.230 tarefas e 24.600 condições de GNN. O objetivo era descobrir se propriedades do grafo original conseguiam prever sucesso antes do rewiring.", st["body"]))
    syn_rows = [
        ["Pipeline selecionado - original", "-4,379 p.p.", "6,34%"],
        ["Delaunay puro - original", "-9,139 p.p.", "2,93%"],
        ["VGAE - Delaunay puro", "+4,907 p.p.", "96,10%"],
    ]
    story.append(data_table(["Comparação", "Ganho médio", "Fração positiva"], syn_rows, [8.0 * cm, 4.1 * cm, 4.5 * cm], st, font_size=8.4, header_color=TEAL))
    story.append(charts["synthetic"])
    story.append(heading("10.1 O principal aprendizado", st["h2"], 1))
    story.append(p("O VGAE quase sempre melhorou o Delaunay, mas principalmente reparou danos causados pela reconstrução geométrica. Em média, o reparo não recuperou toda a vantagem do grafo original. Random regular foi a única família com ganho médio positivo, ainda com amostra pequena.", st["body"]))
    story.append(heading("10.2 Por que os sintéticos pareciam contradizer as bases reais", st["h2"], 1))
    story.append(p("No primeiro desenho sintético, o grafo original já era uma topologia válida para a tarefa. Substituí-lo favorecia naturalmente o baseline. Em várias bases reais, a topologia pode ser incompleta ou pouco alinhada com as features. Essa diferença motivou o segundo desenho, com corrupção explícita.", st["body"]))

    # 11 recovery
    story.append(PageBreak())
    story.append(heading("11. Experimento de corrupção e recuperação", st["h1"], 0))
    story.append(p("Nesse experimento, labels e features vieram de uma estrutura latente e o grafo observado foi corrompido depois. Assim, havia um defeito real que o rewiring poderia corrigir. Foram 180 grafos, 1.080 execuções aninhadas e 16.200 condições.", st["body"]))
    story.append(charts["recovery"])
    recovery_rows = [
        ["Pipeline learned-feature", "-0,802 p.p.", "42,2%"],
        ["Delaunay puro", "-2,552 p.p.", "-"],
        ["Incremento VGAE sobre Delaunay", "+2,008 p.p.", "Reparo parcial"],
        ["Raw-feature kNN", "+6,910 p.p.", "93,9%"],
        ["Raw-feature UMAP-Delaunay", "+4,420 p.p.", "86,7%"],
    ]
    story.append(data_table(["Condição", "Ganho médio", "Fração positiva / papel"], recovery_rows, [7.5 * cm, 4.0 * cm, 5.1 * cm], st, font_size=8.1))
    story.append(p("As features brutas preservavam o sinal latente. O Auxiliary GCN, treinado sobre o grafo corrompido, absorveu parte da corrupção. Isso mostrou que escolher a representação é tão importante quanto escolher o construtor.", st["body"]))
    story.append(heading("11.1 Gate de validação", st["h2"], 1))
    story.append(p("Foi criado um gate que compara o original e vários candidatos. O rewiring só é aceito quando o melhor candidato supera o original por pelo menos 1 p.p. na validação.", st["body"]))
    gate_primary = gate[gate["aggregation_level"].eq("graph_mean_primary")].iloc[0]
    gate_rows = [
        ["Grafos independentes", f"{int(gate_primary['n'])}"],
        ["Ganho médio de teste", fmt_pp(gate_primary["mean_gain_pp"])],
        ["IC95% bootstrap", f"[{gate_primary['bootstrap_ci95_low_pp']:.3f}; {gate_primary['bootstrap_ci95_high_pp']:.3f}] p.p.".replace(".", ",")],
        ["Grafos com ganho positivo", fmt_pct(100 * gate_primary["positive_fraction"])],
        ["Grafos com ganho acima de 0,5 p.p.", fmt_pct(100 * gate_primary["gain_gt_0_5pp_fraction"])],
    ]
    story.append(data_table(["Indicador", "Resultado"], gate_rows, [9.0 * cm, 7.6 * cm], st, font_size=8.6, header_color=GREEN))
    story.append(callout("O gate é o resultado operacional mais forte: ele não tenta adivinhar uma regra universal. Ele aplica o rewiring somente quando a validação mostra vantagem suficiente.", st["callout"], PALE_GREEN, GREEN))

    # 12 blind
    story.append(PageBreak())
    story.append(heading("12. Indicadores estruturais e validação cega", st["h1"], 0))
    story.append(p("Foram testadas métricas isoladas, regressões lineares, modelos não lineares e regras explícitas com condições E/OU. Depois, previsões foram congeladas para 96 grafos novos antes de revelar o resultado real.", st["body"]))
    blind_rows = [
        ["Grafos cegos", "96"],
        ["Positivos reais", "9"],
        ["Accuracy", "0,646"],
        ["Balanced accuracy", "0,556"],
        ["ROC-AUC", "0,430"],
        ["Precisão", "0,121"],
        ["Sensibilidade", "0,444"],
        ["Matriz de confusão", "TN 58, FP 29, FN 5, TP 4"],
    ]
    story.append(data_table(["Métrica", "Resultado"], blind_rows, [8.2 * cm, 8.4 * cm], st, font_size=8.5))
    story.append(p("A accuracy isolada parece aceitável porque quase todos os casos eram negativos. Balanced accuracy, AUC e precisão mostram que a regra não generalizou. O resultado cientificamente útil é negativo: topologia original sozinha não basta para decidir.", st["body"]))
    story.append(heading("12.1 O que ainda pode ser usado como indicador", st["h2"], 1))
    story.append(p("Densidade, lambda2, degree CV, homofilia, resistência e conectividade continuam sendo sinais exploratórios e explicativos. Eles podem compor uma análise, mas não devem substituir a validação do candidato. O melhor indicador disponível é o desempenho de validação obtido nas mesmas condições.", st["body"]))

    # 13 latest nested
    story.append(PageBreak())
    story.append(heading("13. Teste mais recente: Auxiliary GCN e validações separadas", st["h1"], 0))
    story.append(p("Este foi o teste final solicitado para responder duas dúvidas específicas: o ganho viria apenas do encoder supervisionado? A reutilização da mesma validação poderia favorecer a escolha da topologia?", st["body"]))
    story.append(heading("13.1 Escopo", st["h2"], 1))
    story += bullets([
        "Datasets: Airports-USA, Cora, Amazon-Photo e Roman-Empire.",
        "Backbones downstream: GCN, GAT e GraphSAGE.",
        "10 seeds por combinação, totalizando 120 comparações pareadas.",
        "Três condições: GNN original, Auxiliary GCN sozinho e reconstrução seguida da GNN downstream.",
        "Split novo: 60% treino, 10% inner-validation, 10% topology-validation e 20% teste.",
    ], st["bullet"])
    story.append(charts["nested"])

    nested_means = nested_selected.groupby("dataset", as_index=False).agg(
        original=("original_test_f1", "mean"),
        auxiliary=("auxiliary_test_f1", "mean"),
        reconstructed=("selected_test_f1", "mean"),
        gain=("gain_selected_vs_original", "mean"),
    )
    latest_rows = []
    for row in nested_means.itertuples():
        latest_rows.append([
            row.dataset,
            fmt_pct(100 * row.original, 2),
            fmt_pct(100 * row.auxiliary, 2),
            fmt_pct(100 * row.reconstructed, 2),
            fmt_pp(100 * row.gain),
        ])
    story.append(data_table(
        ["Dataset", "Original", "Auxiliary only", "Reconstruído", "Ganho vs original"],
        latest_rows,
        [4.1 * cm, 3.0 * cm, 3.4 * cm, 3.3 * cm, 3.4 * cm],
        st,
        font_size=7.8,
        header_color=TEAL,
    ))
    story.append(heading("13.2 Resultado positivo principal", st["h2"], 1))
    roman = nested_gain[nested_gain["dataset"].eq("Roman-Empire")].set_index("backbone")
    story += bullets([
        f"Roman-Empire + GAT: {fmt_pp(roman.loc['gat', 'mean_gain_pp'])}, positivo em {fmt_pct(roman.loc['gat', 'positive_runs_pct'], 0)} das seeds, Wilcoxon p = {roman.loc['gat', 'wilcoxon_p_value']:.3f}.",
        f"Roman-Empire + GCN: {fmt_pp(roman.loc['gcn', 'mean_gain_pp'])}, positivo em {fmt_pct(roman.loc['gcn', 'positive_runs_pct'], 0)} das seeds, Wilcoxon p = {roman.loc['gcn', 'wilcoxon_p_value']:.3f}.",
        "O Auxiliary GCN sozinho obteve aproximadamente 2,68% de macro-F1 em Roman-Empire, muito abaixo das GNNs downstream reconstruídas.",
    ], st["bullet"])
    story.append(callout(
        "Resposta direta: os ganhos de Roman-Empire não vêm apenas do fato de o encoder supervisionado já ter aprendido a tarefa. O Auxiliary GCN sozinho praticamente falhou, enquanto a representação usada para reconstruir o grafo permitiu ganhos com GAT e GCN.",
        st["callout"], PALE_GREEN, GREEN,
    ))
    story.append(heading("13.3 O que mudou com a validação separada", st["h2"], 1))
    story.append(p("Os ganhos em Roman-Empire com GAT e GCN permaneceram depois de separar checkpoint, escolha de topologia e teste. Isso fortalece a interpretação de que existe um efeito real em combinações específicas. Airports-USA ficou levemente positivo com GCN e GraphSAGE, porém os intervalos cruzaram zero; o ganho grande do protocolo anterior não se repetiu nesse piloto. Cora e Amazon-Photo permaneceram negativos.", st["body"]))
    story.append(p("Essa diferença não invalida o estudo anterior. O novo piloto mudou o tamanho da validação, reconstruiu a geometria por seed e usou 10 seeds em vez de 100. Ele mostra que a estimativa de ganho é sensível ao desenho de seleção, justamente a razão de separar as validações.", st["body"]))

    # 14 conclusion thesis
    story.append(PageBreak())
    story.append(heading("14. Conclusões consolidadas", st["h1"], 0))
    conclusions = [
        ["Há ganho real?", "Sim, em combinações específicas. Roman-Empire + GAT é a evidência recente mais limpa; Airports e outros datasets tiveram ganhos fortes no protocolo real anterior."],
        ["O ganho é apenas o Auxiliary GCN?", "Não. Em Roman-Empire, o Auxiliary GCN sozinho teve desempenho muito baixo, mas o grafo reconstruído melhorou GAT e GCN."],
        ["Delaunay é a única causa?", "Não. kNN e mutual-kNN também funcionaram. A representação informativa é parte essencial."],
        ["VGAE ajuda?", "Sim, principalmente reparando arestas removidas pela geometria. Ele melhora Delaunay de forma consistente nos sintéticos."],
        ["Uma métrica estrutural decide quando usar?", "Não com confiabilidade. As regras topology-only falharam na validação cega."],
        ["Como escolher na prática?", "Comparar um conjunto pequeno de candidatos com o original e aceitar rewiring somente quando a validação superar uma margem pré-fixada."],
        ["O backbone importa?", "Muito. GCN e GAT foram melhores em média; GraphSAGE apresentou perdas fortes em alguns grafos."],
    ]
    story.append(data_table(["Pergunta", "Resposta baseada nos experimentos"], conclusions, [5.0 * cm, 11.6 * cm], st, font_size=8.1, header_color=NAVY))
    story.append(heading("14.1 Tese final recomendada", st["h2"], 1))
    story.append(callout(
        "O rewiring é útil quando a representação usada para reconstruir vizinhanças preserva relações discriminativas que o grafo observado expressa de forma incompleta, enquanto o construtor e o refinamento mantêm conectividade suficiente para o message passing. A geometria propõe relações locais; a similaridade repara conexões úteis; e a validação impede substituir um grafo original já adequado.",
        st["callout"], PALE_BLUE, BLUE,
    ))
    story.append(heading("14.2 Contribuições defendíveis para o paper", st["h2"], 1))
    story += bullets([
        "Avaliação protocol-matched em 12 datasets, três backbones e métodos SOTA.",
        "Separação experimental entre representação, construtor, refinamento e aleatoriedade.",
        "Evidência de que VGAE atua como reparador da geometria local.",
        "Demonstração de que métricas topológicas isoladas não predizem adequação de forma confiável.",
        "Gate de validação como alternativa prática a uma regra estrutural fixa.",
        "Controle final mostrando que o ganho observado não é apenas o classificador Auxiliary GCN.",
        "Protocolo final com inner-validation e topology-validation distintas.",
    ], st["bullet"])

    # 15 decision guide
    story.append(PageBreak())
    story.append(heading("15. Guia prático de decisão", st["h1"], 0))
    decision_rows = [
        ["1", "Treinar o original", "Obter baseline de validação e custo."],
        ["2", "Escolher poucos candidatos", "Priorizar raw kNN, raw UMAP-Delaunay, learned kNN e o híbrido quando o orçamento permitir."],
        ["3", "Fixar splits e seeds", "Garantir comparação pareada."],
        ["4", "Separar validações", "Checkpoint na inner-validation; topologia na topology-validation."],
        ["5", "Aplicar margem", "Aceitar rewiring somente se superar o original por 1 ou 2 p.p. na validação."],
        ["6", "Verificar consistência", "Evitar decisões apoiadas em uma única seed."],
        ["7", "Consultar o teste uma vez", "Avaliar a decisão já congelada."],
        ["8", "Registrar custo", "Tempo, RAM, VRAM, tuning e número de candidatos."],
    ]
    story.append(data_table(["Passo", "Ação", "Motivo"], decision_rows, [1.5 * cm, 5.1 * cm, 10.0 * cm], st, font_size=8.1, header_color=GREEN))
    story.append(heading("15.1 Quando evitar a reconstrução", st["h2"], 1))
    story += bullets([
        "Quando nenhum candidato supera a margem de validação.",
        "Quando o grafo original já produz resultado alto e estável e o custo não compensa o ganho pequeno.",
        "Quando a reconstrução fragmenta o grafo ou reduz demais a conectividade.",
        "Quando o backbone escolhido mostrou interação negativa, como GraphSAGE em Roman-Empire no piloto final.",
        "Quando a representação usada para construir vizinhos absorve ruído ou corrupção do grafo observado.",
    ], st["bullet"])
    story.append(heading("15.2 O papel do custo", st["h2"], 1))
    story.append(p("Custo deve entrar na decisão porque SDRF, DiffWire, Auxiliary GCN, UMAP e VGAE não têm o mesmo tempo. Raw-feature kNN é atraente por evitar várias etapas. Os experimentos registraram tempos em parte das execuções, mas não instrumentaram wall-clock, RAM e VRAM de todos os métodos de forma uniforme. Portanto, a conclusão segura é de potencial menor complexidade operacional, não uma vantagem quantitativa fechada.", st["body"]))

    # 16 strengths limitations stop rule
    story.append(PageBreak())
    story.append(heading("16. Pontos fortes, limites e regra de parada", st["h1"], 0))
    story.append(heading("16.1 Pontos fortes", st["h2"], 1))
    story += bullets([
        "Amplitude: bases reais, controles SOTA, construtores, métricas estruturais e sintéticos.",
        "Rastreabilidade: resultados por seed e seleção exclusivamente por validação.",
        "Comparações pareadas, intervalos bootstrap e Wilcoxon.",
        "Validação cega com previsões congeladas antes do gabarito.",
        "Resultados negativos mantidos e usados para reformular a hipótese.",
        "Teste final direcionado às duas maiores dúvidas metodológicas restantes.",
    ], st["bullet"])
    story.append(heading("16.2 Limitações que devem constar no texto", st["h2"], 1))
    story += bullets([
        "As correlações estruturais iniciais usam somente 12 datasets.",
        "Os sintéticos não cobrem toda a diversidade dos grafos reais.",
        "Os indicadores cegos tiveram poucos casos positivos.",
        "O gate de 1 p.p. foi explorado no mesmo estudo em que foi medido.",
        "O piloto final usou quatro datasets e 10 seeds, com hiperparâmetros reaproveitados do estudo anterior.",
        "Custo computacional ainda não foi medido de forma uniforme entre todos os métodos.",
    ], st["bullet"])
    story.append(heading("16.3 Por que não é necessário testar infinitamente", st["h2"], 1))
    story.append(p("Os testes atuais já respondem às perguntas centrais: existem ganhos condicionais, o efeito não é apenas o Auxiliary GCN, Delaunay não é a única opção, o VGAE repara a geometria e regras topology-only não bastam. Novos testes devem existir apenas se forem necessários para uma alegação específica do paper.", st["body"]))
    stop_rows = [
        ["Alegação", "Evidência atual", "Situação"],
        ["Há ganho em combinações específicas", "Bases reais e Roman-Empire no protocolo separado", "Sustentada"],
        ["O ganho não é só o classificador auxiliar", "Auxiliary only muito baixo em Roman-Empire; downstream reconstruído positivo", "Sustentada"],
        ["Delaunay é sempre o melhor", "kNN e mutual-kNN competitivos", "Não sustentada e não necessária"],
        ["Uma métrica topológica prevê sucesso", "Validação cega próxima do acaso", "Não sustentada"],
        ["Gate funciona fora da amostra", "Resultado forte, mas margem exploratória", "Única confirmação futura prioritária"],
        ["Método é mais rápido que SOTA", "Instrumentação incompleta", "Medir somente se essa alegação entrar no paper"],
    ]
    story.append(data_table(stop_rows[0], stop_rows[1:], [6.0 * cm, 7.0 * cm, 3.6 * cm], st, font_size=7.8, header_color=ORANGE))

    # 17 paper framing
    story.append(PageBreak())
    story.append(heading("17. Enquadramento recomendado para publicação", st["h1"], 0))
    story.append(p("A narrativa mais forte não é uma competição para provar que um único construtor vence sempre. É uma investigação sobre quando reconstruções task-informed ajudam, quais componentes produzem ou reparam o efeito e por que uma regra estática baseada somente na topologia falha.", st["body"]))
    story.append(heading("17.1 Pergunta central sugerida", st["h2"], 1))
    story.append(callout("Quando representações task-informed e construtores locais produzem rewiring útil, e por que métricas topológicas do grafo original não são suficientes para prever esse benefício?", st["callout"]))
    story.append(heading("17.2 Estrutura possível do artigo", st["h2"], 1))
    paper_rows = [
        ["Introdução", "Problema de topologias imperfeitas e custo de rewiring inadequado."],
        ["Método", "Auxiliary GCN, UMAP, construtor, VGAE e downstream GNN."],
        ["Protocolo", "Splits, seeds, validação, seleção, métricas e testes pareados."],
        ["Resultados reais", "12 datasets, três backbones e heterogeneidade do ganho."],
        ["Controles", "Raw features, construtores, aleatórios, SDRF e DiffWire."],
        ["Mecanismo", "Before/after estrutural, sintéticos e papel reparador do VGAE."],
        ["Aplicabilidade", "Falha do indicador topology-only e desempenho do gate."],
        ["Validação final", "Auxiliary only e validações separadas."],
        ["Discussão", "Representação, conectividade, backbone, custo e limites."],
    ]
    story.append(data_table(["Seção", "Conteúdo"], paper_rows, [4.2 * cm, 12.4 * cm], st, font_size=8.2))

    # 18 reproducibility
    story.append(PageBreak())
    story.append(heading("18. Reprodutibilidade e arquivos de evidência", st["h1"], 0))
    story.append(p("Os números desta apostila vêm dos artefatos consolidados abaixo. CSVs long format preservam candidatos e seeds; CSVs wide preservam a seleção final pareada; relatórios Markdown registram decisões e interpretação.", st["body"]))
    evidence = [
        ["Bases reais", "global_12datasets_corrected_20260824/final_selected_test_results.csv"],
        ["Testes pareados", "global_12datasets_corrected_20260824/paired_statistical_tests.csv"],
        ["Estrutura", "global_12datasets_corrected_20260824/structural_metric_correlations.csv"],
        ["SOTA", "global_protocol_matched_sota_controls_N30_lean/sota_summary.csv"],
        ["Construtores", "global_protocol_matched_sota_controls_N30_lean/constructor_summary.csv"],
        ["Sintéticos", "synthetic_suitability_pilot_20260917/graph_level_outcomes.csv"],
        ["Corrupção", "corruption_recovery_pilot_20260918/selected_execution_results.csv"],
        ["Gate", "corruption_recovery_pilot_20260918/validation_gate_1pp_per_seed_wide.csv"],
        ["Cego", "blind_sequential_20260919/blind_all_frozen_predictions_scored.csv"],
        ["Teste final", "nested_validation_auxiliary_pilot_20260920/nested_auxiliary_selected_per_seed.csv"],
        ["Candidatos finais", "nested_validation_auxiliary_pilot_20260920/nested_auxiliary_candidates_long.csv"],
    ]
    story.append(data_table(["Bloco", "Arquivo principal"], evidence, [4.0 * cm, 12.6 * cm], st, font_size=7.5))
    story.append(heading("18.1 Checklist para repetir um resultado", st["h2"], 1))
    story += bullets([
        "Fixar dataset, versão, backbone, seed e split.",
        "Registrar todos os candidatos antes de olhar o teste.",
        "Selecionar checkpoint na inner-validation e topologia na topology-validation.",
        "Salvar val_f1 e test_f1 de cada candidato em long format.",
        "Salvar uma linha final por dataset, backbone e seed.",
        "Calcular ganho pareado, bootstrap, Wilcoxon e fração de runs positivas.",
        "Registrar tempo, RAM e VRAM se custo fizer parte da alegação.",
    ], st["bullet"])

    # 19 glossary
    story.append(PageBreak())
    story.append(heading("19. Glossário rápido", st["h1"], 0))
    glossary = [
        ["Baseline", "Resultado de referência, normalmente a GNN no grafo original."],
        ["Candidate", "Uma topologia ou taxa que concorre na validação."],
        ["Checkpoint", "Estado do modelo salvo na época com melhor validação."],
        ["Encoder", "Modelo que transforma features em uma representação aprendida."],
        ["Early stopping", "Escolha do momento de parar com base na validação."],
        ["Gate", "Regra que decide manter o original ou aceitar um candidato."],
        ["Heterofilia", "Tendência de arestas conectarem classes diferentes."],
        ["Homofilia", "Tendência de arestas conectarem a mesma classe."],
        ["Long format", "Uma linha para cada candidato e execução."],
        ["Out-of-sample", "Avaliação em dados não usados para ajustar a regra."],
        ["Protocol-matched", "Comparação com mesmas condições experimentais."],
        ["Seed", "Número que controla aleatoriedade e torna a execução reproduzível."],
        ["Topology-validation", "Conjunto usado somente para escolher o grafo."],
        ["Wide format", "Uma linha por seed com original, selecionado e decisão."],
    ]
    story.append(data_table(["Termo", "Significado"], glossary, [4.2 * cm, 12.4 * cm], st, font_size=8.1))

    # Closing
    story.append(PageBreak())
    story.append(heading("20. Síntese final em uma página", st["h1"], 0))
    story.append(callout("O método funciona quando a reconstrução encontra relações úteis que o grafo original não expressa bem e preserva conectividade suficiente para a GNN downstream.", st["callout"], PALE_GREEN, GREEN))
    story += [Spacer(1, 0.25 * cm)]
    final_points = [
        "Os 12 datasets reais mostraram ganhos importantes, mas dependentes do dataset e do backbone.",
        "Controles provaram que a representação aprendida contém sinal; Delaunay não é o único construtor válido.",
        "VGAE atua principalmente como reparo das perdas criadas pela geometria local.",
        "Features brutas podem ser superiores quando o encoder aprende sobre um grafo corrompido.",
        "Métricas topológicas ajudam a explicar, mas não decidiram bem em validação cega.",
        "O gate de validação com margem de 1 p.p. foi o melhor resultado operacional: +5,398 p.p. em média por grafo.",
        "O teste final separou treino, checkpoint, escolha da topologia e teste.",
        "Roman-Empire + GAT ganhou +1,83 p.p. nas 10 seeds; GCN ganhou +1,34 p.p. em média.",
        "O Auxiliary GCN sozinho praticamente falhou em Roman-Empire, comprovando que esses ganhos não são apenas o encoder classificando diretamente.",
        "A próxima prioridade não é testar infinitamente: é confirmar o gate fora da amostra e medir custo apenas se essas alegações entrarem no artigo.",
    ]
    story += bullets(final_points, st["bullet"])
    story += [Spacer(1, 0.4 * cm), HRFlowable(width="100%", thickness=2, color=TEAL), Spacer(1, 0.3 * cm)]
    story.append(p("Conclusão: a pesquisa evoluiu de uma hipótese ampla sobre Delaunay para uma tese mais forte e mais precisa sobre representação, reconstrução, reparo, backbone e seleção por validação.", st["callout"]))
    return story


def main() -> None:
    register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_prepare()
    charts = build_charts(data)
    st = styles()
    doc = BookletDoc(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.45 * cm,
        title="Apostila completa do estudo de rewiring geométrico e híbrido",
        author="Projeto VGAE-Delaunay Hybrid Rewiring",
        subject="Metodologia, métricas, resultados, decisões e conclusões",
    )
    story = build_story(st, data, charts)
    doc.multiBuild(story)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
