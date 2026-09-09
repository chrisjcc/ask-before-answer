"""
Shared styling module for AskBeforeAnswer TRD / TDD documents.
Visual language modeled after TRD_CreditFraud_Pipeline.pdf:
 - dark navy title block, key/value metadata table
 - navy numbered H1 sections, blue underlined H2 subsections
 - navy-header / gray-banded data tables
 - tan/orange NOTE and IMPORTANT callout boxes
 - light running header + footer with classification and page number
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, NextPageTemplate, PageBreak, HRFlowable, ListFlowable, ListItem
)
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------- palette --
NAVY = colors.HexColor("#1F3864")
NAVY_DARK = colors.HexColor("#16294A")
BLUE = colors.HexColor("#2E74B5")
BLUE_LIGHT = colors.HexColor("#DCE6F1")
GRAY_BAND = colors.HexColor("#F2F2F2")
GRAY_BORDER = colors.HexColor("#BFBFBF")
GRAY_TEXT = colors.HexColor("#595959")
TAN_BG = colors.HexColor("#FCE9C8")
TAN_BORDER = colors.HexColor("#E8B96E")
ORANGE_TEXT = colors.HexColor("#B15C00")
GREEN_OK = colors.HexColor("#E2EFDA")
GREEN_TEXT = colors.HexColor("#548235")
CODE_BG = colors.HexColor("#F4F4F4")
CODE_BORDER = colors.HexColor("#D9D9D9")
WHITE = colors.white
BLACK_TEXT = colors.HexColor("#1A1A1A")

PAGE_W, PAGE_H = letter
MARGIN_L = 0.85 * inch
MARGIN_R = 0.85 * inch
MARGIN_T = 0.95 * inch
MARGIN_B = 0.85 * inch

# ----------------------------------------------------------------- styles --
_ss = getSampleStyleSheet()

styles = {}
styles["Body"] = ParagraphStyle(
    "Body", parent=_ss["Normal"], fontName="Helvetica", fontSize=9.6,
    leading=14, spaceAfter=8, textColor=BLACK_TEXT, alignment=TA_LEFT,
)
styles["BodyBold"] = ParagraphStyle(
    "BodyBold", parent=styles["Body"], fontName="Helvetica-Bold",
)
styles["DocTitle"] = ParagraphStyle(
    "DocTitle", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=25,
    leading=29, textColor=NAVY, spaceAfter=6,
)
styles["DocSubtitle"] = ParagraphStyle(
    "DocSubtitle", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=14.5,
    leading=18, textColor=BLUE, spaceAfter=14,
)
styles["Classification"] = ParagraphStyle(
    "Classification", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=9.5,
    leading=12, textColor=ORANGE_TEXT, spaceBefore=4, spaceAfter=16,
)
styles["H1"] = ParagraphStyle(
    "H1", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=15.5,
    leading=19, textColor=NAVY, spaceBefore=18, spaceAfter=10,
)
styles["H2"] = ParagraphStyle(
    "H2", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=11.8,
    leading=15, textColor=BLUE, spaceBefore=12, spaceAfter=6,
)
styles["H3"] = ParagraphStyle(
    "H3", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=10.3,
    leading=13, textColor=NAVY_DARK, spaceBefore=8, spaceAfter=4,
)
styles["Bullet"] = ParagraphStyle(
    "Bullet", parent=styles["Body"], leftIndent=14, bulletIndent=2, spaceAfter=4,
)
styles["TableCell"] = ParagraphStyle(
    "TableCell", parent=_ss["Normal"], fontName="Helvetica", fontSize=8.7,
    leading=11.5, textColor=BLACK_TEXT,
)
styles["TableCellBold"] = ParagraphStyle(
    "TableCellBold", parent=styles["TableCell"], fontName="Helvetica-Bold",
)
styles["TableHeader"] = ParagraphStyle(
    "TableHeader", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=8.9,
    leading=11.5, textColor=WHITE,
)
styles["MetaLabel"] = ParagraphStyle(
    "MetaLabel", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=9.3,
    leading=12.5, textColor=WHITE,
)
styles["MetaValue"] = ParagraphStyle(
    "MetaValue", parent=_ss["Normal"], fontName="Helvetica", fontSize=9.3,
    leading=12.5, textColor=BLACK_TEXT,
)
styles["MetaValueStatus"] = ParagraphStyle(
    "MetaValueStatus", parent=styles["MetaValue"], fontName="Helvetica-Bold",
    textColor=ORANGE_TEXT,
)
styles["Code"] = ParagraphStyle(
    "Code", parent=_ss["Normal"], fontName="Courier", fontSize=8.3,
    leading=11.5, textColor=BLACK_TEXT,
)
styles["NoteLabel"] = ParagraphStyle(
    "NoteLabel", parent=_ss["Normal"], fontName="Helvetica-Bold", fontSize=8.6,
    leading=11, textColor=ORANGE_TEXT,
)
styles["NoteBody"] = ParagraphStyle(
    "NoteBody", parent=_ss["Normal"], fontName="Helvetica", fontSize=8.9,
    leading=12.2, textColor=BLACK_TEXT,
)
styles["Footer"] = ParagraphStyle(
    "Footer", parent=_ss["Normal"], fontName="Helvetica", fontSize=7.6,
    leading=9, textColor=GRAY_TEXT,
)
styles["RunningHead"] = ParagraphStyle(
    "RunningHead", parent=_ss["Normal"], fontName="Helvetica", fontSize=8.2,
    leading=10, textColor=GRAY_TEXT,
)


def P(text, style="Body"):
    return Paragraph(text, styles[style])


def spacer(h=8):
    return Spacer(1, h)


def hrule(color=BLUE, thickness=1.1, width="100%", space_before=2, space_after=10):
    return HRFlowable(width=width, thickness=thickness, color=color,
                       spaceBefore=space_before, spaceAfter=space_after)


def h1(number, text):
    return P(f"{number}. {text}", "H1")


def h2(text):
    return KeepTogether([P(text, "H2"), hrule(color=BLUE, thickness=0.9, space_before=0, space_after=8)])


def h3(text):
    return P(text, "H3")


def bullets(items, style="Bullet"):
    flows = []
    for it in items:
        flows.append(Paragraph(f"&bull;&nbsp;&nbsp;{it}", styles[style]))
    return flows


def numbered(items, style="Bullet"):
    flows = []
    for i, it in enumerate(items, 1):
        flows.append(Paragraph(f"{i}.&nbsp;&nbsp;{it}", styles[style]))
    return flows


def note_box(label, text, bg=TAN_BG, border=TAN_BORDER, label_style="NoteLabel"):
    inner = Table(
        [[Paragraph(f"{label}", styles[label_style]), Paragraph(text, styles["NoteBody"])]],
        colWidths=[0.85 * inch, None],
    )
    inner.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.75, border),
        ("LEFTPADDING", (0, 0), (0, 0), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 10),
    ]))
    return KeepTogether([spacer(4), inner, spacer(10)])


def data_table(header_row, rows, col_widths=None, header_bg=NAVY, band=True,
               font_size=8.7, align_first_left=True):
    """rows: list of list-of-strings (converted to Paragraphs)."""
    def cell(v, bold=False, header=False):
        if header:
            return Paragraph(str(v), styles["TableHeader"])
        st = styles["TableCellBold"] if bold else styles["TableCell"]
        return Paragraph(str(v), st)

    data = [[cell(h, header=True) for h in header_row]]
    for r in rows:
        data.append([cell(v) for v in r])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, header_bg),
        ("BOX", (0, 0), (-1, -1), 0.6, GRAY_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, GRAY_BORDER),
    ]
    if band:
        for i in range(1, len(data)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), GRAY_BAND))
    t.setStyle(TableStyle(style_cmds))
    return t


def code_block(lines):
    text = "<br/>".join(
        l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
        for l in lines
    )
    tbl = Table([[Paragraph(text, styles["Code"])]], colWidths=[None])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, CODE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return KeepTogether([tbl, spacer(10)])


def diagram_block(lines, align=TA_CENTER):
    """Monospace ASCII-art flow diagram inside a bordered box."""
    dstyle = ParagraphStyle(
        "Diagram", parent=styles["Code"], alignment=align, leading=12.5, fontSize=8.6,
    )
    text = "<br/>".join(
        l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
        for l in lines
    )
    tbl = Table([[Paragraph(text, dstyle)]], colWidths=[None])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, CODE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return KeepTogether([tbl, spacer(10)])


def meta_table(rows):
    """rows: list of (label, value, is_status_bool)"""
    data = []
    for label, value, is_status in rows:
        vstyle = "MetaValueStatus" if is_status else "MetaValue"
        data.append([Paragraph(label, styles["MetaLabel"]), Paragraph(value, styles[vstyle])])
    t = Table(data, colWidths=[1.5 * inch, 4.85 * inch])
    cmds = [
        ("BACKGROUND", (0, 0), (0, -1), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.6, GRAY_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(len(rows)):
        if i % 2 == 1:
            cmds.append(("BACKGROUND", (1, i), (1, i), GRAY_BAND))
        else:
            cmds.append(("BACKGROUND", (1, i), (1, i), WHITE))
    t.setStyle(TableStyle(cmds))
    return t


# ----------------------------------------------------------- page frames --
def _header_footer(canvas, doc, running_title, footer_left):
    canvas.saveState()
    # running header
    canvas.setFont("Helvetica", 8.2)
    canvas.setFillColor(GRAY_TEXT)
    canvas.drawString(MARGIN_L, PAGE_H - 0.62 * inch, running_title)
    canvas.setStrokeColor(GRAY_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_L, PAGE_H - 0.68 * inch, PAGE_W - MARGIN_R, PAGE_H - 0.68 * inch)
    # footer
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(GRAY_TEXT)
    canvas.drawString(MARGIN_L, 0.55 * inch, footer_left)
    canvas.drawRightString(PAGE_W - MARGIN_R, 0.55 * inch, f"Page {doc.page}")
    canvas.setStrokeColor(GRAY_BORDER)
    canvas.line(MARGIN_L, 0.66 * inch, PAGE_W - MARGIN_R, 0.66 * inch)
    canvas.restoreState()


def build_doc(filepath, running_title, footer_left, story):
    doc = BaseDocTemplate(
        filepath, pagesize=letter,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T, bottomMargin=MARGIN_B,
        title=running_title,
    )
    frame = Frame(MARGIN_L, MARGIN_B, PAGE_W - MARGIN_L - MARGIN_R,
                   PAGE_H - MARGIN_T - MARGIN_B, id="normal")

    def on_page(canvas, doc_):
        _header_footer(canvas, doc_, running_title, footer_left)

    template = PageTemplate(id="main", frames=[frame], onPage=on_page)
    doc.addPageTemplates([template])
    doc.build(story)
