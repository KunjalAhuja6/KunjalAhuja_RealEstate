"""
PDF Documentation Generator for Real Estate Valuation Assistant
Author / Presenter: Kunjal Ahuja

Generates clean, professional PDF documentation for:
1. design/Design_Document.pdf
2. docs/Dataset_Details.pdf
3. docs/API_Documentation.pdf
4. docs/Evaluation_Report.pdf
"""

import os
import re
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw 'Page X of Y'."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, num_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, getattr(self, "doc_title", "Real Estate Valuation Assistant"))
            self.drawRightString(558, 750, "Kunjal Ahuja • Capstone Project")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
        
        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "AI-Powered Real Estate Property Valuation Assistant")
        page_str = f"Page {self._pageNumber} of {num_pages}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()


def get_styles():
    base = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle', parent=base['Heading1'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        textColor=colors.HexColor('#1e3a8a'), spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=base['Normal'],
        fontName='Helvetica', fontSize=10, leading=14,
        textColor=colors.HexColor('#475569'), spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'DocH1', parent=base['Heading1'],
        fontName='Helvetica-Bold', fontSize=14, leading=18,
        textColor=colors.HexColor('#1e3a8a'), spaceBefore=14, spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'DocH2', parent=base['Heading2'],
        fontName='Helvetica-Bold', fontSize=11, leading=15,
        textColor=colors.HexColor('#0f766e'), spaceBefore=10, spaceAfter=5,
        keepWithNext=True
    )
    h3_style = ParagraphStyle(
        'DocH3', parent=base['Heading3'],
        fontName='Helvetica-Bold', fontSize=9.5, leading=13,
        textColor=colors.HexColor('#1e293b'), spaceBefore=8, spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'DocBody', parent=base['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=colors.HexColor('#1e293b'), spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'DocBullet', parent=base['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=colors.HexColor('#1e293b'), leftIndent=14, firstLineIndent=-10, spaceAfter=3
    )
    code_style = ParagraphStyle(
        'DocCode', parent=base['Normal'],
        fontName='Courier', fontSize=7.5, leading=10,
        textColor=colors.HexColor('#0f172a'), spaceAfter=6
    )
    table_cell = ParagraphStyle(
        'TableCell', parent=base['Normal'],
        fontName='Helvetica', fontSize=7.5, leading=10,
        textColor=colors.HexColor('#1e293b')
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold', parent=base['Normal'],
        fontName='Helvetica-Bold', fontSize=7.5, leading=10,
        textColor=colors.HexColor('#1e293b')
    )
    table_cell_head = ParagraphStyle(
        'TableCellHead', parent=base['Normal'],
        fontName='Helvetica-Bold', fontSize=7.5, leading=10,
        textColor=colors.white
    )
    callout_style = ParagraphStyle(
        'DocCallout', parent=base['Normal'],
        fontName='Helvetica', fontSize=8.0, leading=11.5,
        textColor=colors.HexColor('#0f766e')
    )
    
    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'h1': h1_style,
        'h2': h2_style,
        'h3': h3_style,
        'body': body_style,
        'bullet': bullet_style,
        'code': code_style,
        'table_cell': table_cell,
        'table_cell_bold': table_cell_bold,
        'table_cell_head': table_cell_head,
        'callout': callout_style
    }


def clean_math_markdown(text):
    """Replaces raw LaTeX syntax with clean plain-text unicode equivalents."""
    # Replace math blocks $$ ... $$
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: m.group(1).strip(), text, flags=re.DOTALL)
    # Replace inline math $ ... $
    # Be careful not to replace literal dollars like $160,000 or $12,789
    # A math dollar usually has letters, math operators, or backslashes
    def replace_math_dollar(match):
        inner = match.group(1)
        # If it's just numbers with commas/periods, it was a dollar amount! Keep it!
        if re.match(r'^\d[\d,.]*$', inner):
            return f"${inner}"
        # Convert LaTeX symbols
        inner = inner.replace('\\text', '').replace('{', '').replace('}', '')
        inner = inner.replace('\\alpha', 'alpha')
        inner = inner.replace('\\le', '<=')
        inner = inner.replace('\\ge', '>=')
        inner = inner.replace('\\times', 'x')
        inner = inner.replace('\\cdot', '*')
        inner = inner.replace('\\approx', '~')
        inner = inner.replace('\\pm', '+/-')
        inner = inner.replace('\\sum_{i=1}^N', 'SUM')
        inner = re.sub(r'\\min\((.*?)\)', r'min(\1)', inner)
        inner = re.sub(r'\\max\((.*?)\)', r'max(\1)', inner)
        inner = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', inner)
        return inner

    # Matches $...$ where it looks like math (has non-digits or math symbols)
    text = re.sub(r'\$([a-zA-Z\\_][^$]*?)\$', replace_math_dollar, text)
    # Clean up any leftover LaTeX commands
    text = text.replace('\\text', '').replace('\\frac', '').replace('\\le', '<=').replace('\\ge', '>=')
    text = text.replace('\\alpha', 'alpha').replace('\\pm', '+/-').replace('\\times', 'x')
    return text


def escape_xml(text):
    """Escapes XML entities for ReportLab Paragraphs."""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text


def format_inline_markdown(text):
    """Converts bold and code spans to ReportLab tags."""
    text = clean_math_markdown(text)
    # Escape basic xml
    # Protect bold and code before escaping
    # We split tokens
    tokens = []
    # Replace **bold**
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace `code`
    text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
    return text


def parse_markdown_to_story(md_path, styles, doc_width=504):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    story = []
    i = 0
    in_code_block = False
    code_lines = []
    
    while i < len(lines):
        line = lines[i].rstrip('\r\n')
        
        # Code block handling
        if line.startswith('```'):
            if in_code_block:
                in_code_block = False
                code_text = "<br/>".join([escape_xml(cl) for cl in code_lines])
                if code_text:
                    p = Paragraph(f"<b>Code:</b><br/>{code_text}", styles['code'])
                    t = Table([[p]], colWidths=[doc_width])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
                        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                        ('TOPPADDING', (0,0), (-1,-1), 4),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                        ('LEFTPADDING', (0,0), (-1,-1), 6),
                        ('RIGHTPADDING', (0,0), (-1,-1), 6),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 4))
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue
        
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
        
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        
        # Horizontal rule
        if stripped in ('---', '***', '___'):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceBefore=6, spaceAfter=6))
            i += 1
            continue
            
        # Title (# )
        if line.startswith('# '):
            title_text = line[2:].strip()
            title_text = format_inline_markdown(title_text)
            story.append(Paragraph(title_text, styles['title']))
            story.append(Spacer(1, 2))
            i += 1
            continue
            
        # Heading 2 (## )
        if line.startswith('## '):
            h_text = format_inline_markdown(line[3:].strip())
            story.append(Paragraph(h_text, styles['h1']))
            i += 1
            continue
            
        # Heading 3 (### )
        if line.startswith('### '):
            h_text = format_inline_markdown(line[4:].strip())
            story.append(Paragraph(h_text, styles['h2']))
            i += 1
            continue
            
        # Heading 4 (#### )
        if line.startswith('#### '):
            h_text = format_inline_markdown(line[5:].strip())
            story.append(Paragraph(h_text, styles['h3']))
            i += 1
            continue
            
        # Table detection
        if line.startswith('|') and '|' in line[1:]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            
            # Parse table
            if len(table_lines) >= 2:
                rows_data = []
                for tl in table_lines:
                    # Skip separator line |---|---|
                    if re.match(r'^\|[\s\-:|]+\|$', tl):
                        continue
                    cells = [c.strip() for c in tl.strip('|').split('|')]
                    rows_data.append(cells)
                
                if rows_data:
                    num_cols = max(len(r) for r in rows_data)
                    # Normalize columns
                    for r in rows_data:
                        while len(r) < num_cols:
                            r.append('')
                    
                    # Convert cells to Paragraphs
                    table_cells = []
                    for row_idx, r in enumerate(rows_data):
                        row_p = []
                        for col_idx, cell in enumerate(r):
                            cell_text = format_inline_markdown(cell)
                            if row_idx == 0:
                                p = Paragraph(cell_text, styles['table_cell_head'])
                            else:
                                p = Paragraph(cell_text, styles['table_cell'])
                            row_p.append(p)
                        table_cells.append(row_p)
                    
                    # Compute column widths
                    col_w = doc_width / num_cols
                    # Specialized column widths for known tables
                    if num_cols == 4:
                        col_widths = [doc_width*0.22, doc_width*0.28, doc_width*0.22, doc_width*0.28]
                    elif num_cols == 5:
                        col_widths = [doc_width*0.18, doc_width*0.22, doc_width*0.20, doc_width*0.20, doc_width*0.20]
                    elif num_cols == 7:
                        col_widths = [doc_width*0.12, doc_width*0.28, doc_width*0.20, doc_width*0.14, doc_width*0.12, doc_width*0.07, doc_width*0.07]
                    elif num_cols == 6:
                        col_widths = [doc_width*0.16, doc_width*0.20, doc_width*0.16, doc_width*0.16, doc_width*0.16, doc_width*0.16]
                    else:
                        col_widths = [col_w] * num_cols
                    
                    t = Table(table_cells, colWidths=col_widths, repeatRows=1)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('TOPPADDING', (0,0), (-1,-1), 3),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                        ('LEFTPADDING', (0,0), (-1,-1), 4),
                        ('RIGHTPADDING', (0,0), (-1,-1), 4),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 6))
            continue
            
        # Bullet list
        if stripped.startswith('- ') or stripped.startswith('* '):
            bullet_text = format_inline_markdown(stripped[2:].strip())
            story.append(Paragraph(f"• {bullet_text}", styles['bullet']))
            i += 1
            continue
            
        # Numbered list
        num_match = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if num_match:
            num = num_match.group(1)
            item_text = format_inline_markdown(num_match.group(2))
            story.append(Paragraph(f"<b>{num}.</b> {item_text}", styles['bullet']))
            i += 1
            continue
            
        # Blockquote (> )
        if stripped.startswith('> '):
            quote_text = format_inline_markdown(stripped[2:].strip())
            p = Paragraph(quote_text, styles['callout'])
            t = Table([[p]], colWidths=[doc_width])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0fdf4')),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#86efac')),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(t)
            story.append(Spacer(1, 4))
            i += 1
            continue
            
        # Regular paragraph
        p_text = format_inline_markdown(stripped)
        story.append(Paragraph(p_text, styles['body']))
        i += 1
        
    return story


def build_pdf(md_path, pdf_path, title_header):
    print(f"Building {pdf_path} from {md_path}...")
    styles = get_styles()
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54, rightMargin=54,
        topMargin=54, bottomMargin=54
    )
    
    story = parse_markdown_to_story(md_path, styles, doc_width=504)
    
    def on_first_page(canvas, doc):
        canvas.doc_title = title_header
        
    def on_later_pages(canvas, doc):
        canvas.doc_title = title_header
        
    doc.build(story, canvasmaker=NumberedCanvas, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    print(f"Successfully generated {pdf_path} (size: {os.path.getsize(pdf_path)} bytes)")


if __name__ == '__main__':
    docs = [
        ('design/Design_Document.md', 'design/Design_Document.pdf', 'System Design Document'),
        ('docs/Dataset_Details.md', 'docs/Dataset_Details.pdf', 'Dataset Documentation & Details'),
        ('docs/API_Documentation.md', 'docs/API_Documentation.pdf', 'REST API Reference Manual'),
        ('docs/Evaluation_Report.md', 'docs/Evaluation_Report.pdf', 'DeepEval Quality Benchmark Report'),
    ]
    
    for md, pdf, title in docs:
        build_pdf(md, pdf, title)
    print("All documentation PDFs generated successfully.")
