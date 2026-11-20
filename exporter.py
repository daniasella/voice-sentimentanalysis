import io
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_excel(records):
    """
    Menghasilkan file Excel (.xlsx) dari daftar riwayat analisis sentimen.
    Mengembalikan objek io.BytesIO.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Riwayat Sentimen"
    ws.views.sheetView[0].showGridLines = True

    # Palette
    header_fill = PatternFill(start_color="312E81", end_color="312E81", fill_type="solid") # Deep Indigo
    header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Segoe UI", size=16, bold=True, color="1E1B4B")
    sub_font = Font(name="Segoe UI", size=10, italic=True, color="4B5563")
    regular_font = Font(name="Segoe UI", size=10)
    bold_font = Font(name="Segoe UI", size=10, bold=True)
    
    thin_border = Border(
        left=Side(style='thin', color='E2E8F0'),
        right=Side(style='thin', color='E2E8F0'),
        top=Side(style='thin', color='E2E8F0'),
        bottom=Side(style='thin', color='E2E8F0')
    )

    # Title block
    ws.merge_cells("A1:K1")
    ws["A1"] = "LAPORAN RIWAYAT ANALISIS SENTIMEN UCAPAN"
    ws["A1"].font = title_font
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 28

    ws.merge_cells("A2:K2")
    ws["A2"] = f"Waktu Ekspor: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} WIB | Database: Neon PostgreSQL | Total Rekaman: {len(records)}"
    ws["A2"].font = sub_font
    ws["A2"].alignment = Alignment(vertical="center")
    ws.row_dimensions[2].height = 20

    ws.append([]) # row 3 blank

    # Column Headers
    headers = [
        "No",
        "Waktu Rekam",
        "Transkrip Mentah (STT)",
        "Teks Normalisasi",
        "Sentimen STT",
        "Skor Model",
        "Durasi (detik)",
        "Latensi STT (detik)",
        "Teks Koreksi Pengguna",
        "Sentimen Koreksi",
        "Jarak Edit Kata"
    ]
    ws.append(headers) # row 4
    ws.row_dimensions[4].height = 26

    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=4, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    # Data Rows
    for idx, r in enumerate(records, start=1):
        time_str = r.get("created_at_formatted") or "-"
        raw_text = r.get("text_raw") or "-"
        norm_text = r.get("text_normalized") or "-"
        sent = (r.get("sentiment") or "-").upper()
        score = f"{float(r.get('model_score')):.4f}" if r.get("model_score") is not None else "-"
        duration = f"{float(r.get('audio_duration')):.2f}" if r.get("audio_duration") is not None else "-"
        exec_time = f"{float(r.get('execution_time')):.4f}" if r.get("execution_time") is not None else "-"
        corr_text = r.get("user_corrected_text") or "-"
        corr_sent = (r.get("corrected_sentiment") or "-").upper() if r.get("corrected_sentiment") else "-"
        
        edit_dist = "-"
        if r.get("word_edit_distance") is not None:
            ratio = (r.get("word_edit_ratio") or 0.0) * 100
            edit_dist = f"{r.get('word_edit_distance')} kata ({ratio:.1f}%)"

        row_data = [
            idx,
            time_str,
            raw_text,
            norm_text,
            sent,
            score,
            duration,
            exec_time,
            corr_text,
            corr_sent,
            edit_dist
        ]
        ws.append(row_data)
        current_row = 4 + idx
        ws.row_dimensions[current_row].height = 22

        # Style data row
        row_bg = "F8FAFC" if idx % 2 == 0 else "FFFFFF"
        row_fill = PatternFill(start_color=row_bg, end_color=row_bg, fill_type="solid")

        for col_idx in range(1, len(row_data) + 1):
            c = ws.cell(row=current_row, column=col_idx)
            c.font = regular_font
            c.fill = row_fill
            c.border = thin_border

            # Specific alignments
            if col_idx in [1, 2, 6, 7, 8, 11]:
                c.alignment = Alignment(horizontal="center", vertical="center")
            elif col_idx in [5, 10]:
                c.alignment = Alignment(horizontal="center", vertical="center")
                # Highlight sentiment
                s_val = str(c.value).upper()
                if "POSITIF" in s_val:
                    c.font = Font(name="Segoe UI", size=10, bold=True, color="047857")
                elif "NEGATIF" in s_val:
                    c.font = Font(name="Segoe UI", size=10, bold=True, color="B91C1C")
                elif "NETRAL" in s_val:
                    c.font = Font(name="Segoe UI", size=10, bold=True, color="B45309")
            else:
                c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

    # Auto-adjust column widths
    column_widths = {
        1: 6,   # No
        2: 20,  # Waktu
        3: 28,  # Raw
        4: 28,  # Norm
        5: 16,  # Sentimen
        6: 14,  # Skor
        7: 15,  # Durasi
        8: 18,  # Latensi
        9: 28,  # Koreksi
        10: 16, # Sentimen Koreksi
        11: 20  # Jarak Edit
    }
    for col_idx, width in column_widths.items():
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = width

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output

def generate_pdf(records):
    """
    Menghasilkan file PDF Laporan Riwayat Analisis Sentimen (A4 Landscape).
    Mengembalikan objek io.BytesIO.
    """
    buffer = io.BytesIO()
    # A4 landscape: 842 x 595 points
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=28,
        rightMargin=28,
        topMargin=28,
        bottomMargin=28
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1E1B4B'),
        spaceAfter=4
    )
    
    sub_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=12
    )

    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#111827')
    )

    cell_style_center = ParagraphStyle(
        'TableCellCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        alignment=1, # Center
        textColor=colors.HexColor('#111827')
    )

    cell_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.white
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("LAPORAN RIWAYAT ANALISIS SENTIMEN UCAPAN", title_style))
    
    total_count = len(records)
    pos_count = sum(1 for r in records if (r.get("sentiment") or "").lower() == "positif")
    neg_count = sum(1 for r in records if (r.get("sentiment") or "").lower() == "negatif")
    net_count = sum(1 for r in records if (r.get("sentiment") or "").lower() == "netral")
    
    meta_text = (
        f"<b>Tanggal Cetak:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} WIB &nbsp;|&nbsp; "
        f"<b>Database:</b> Neon PostgreSQL &nbsp;|&nbsp; "
        f"<b>Total:</b> {total_count} rekaman &nbsp;|&nbsp; "
        f"<font color='#059669'><b>Positif:</b> {pos_count}</font> &nbsp;|&nbsp; "
        f"<font color='#DC2626'><b>Negatif:</b> {neg_count}</font> &nbsp;|&nbsp; "
        f"<font color='#D97706'><b>Netral:</b> {net_count}</font>"
    )
    elements.append(Paragraph(meta_text, sub_style))

    # Table columns (total printable width around 786 pt)
    col_widths = [24, 65, 150, 150, 60, 48, 48, 52, 100, 55, 34] 
    # sum: 24+65+150+150+60+48+48+52+100+55+34 = 786 pt

    headers = [
        Paragraph("No", cell_header_style),
        Paragraph("Waktu", cell_header_style),
        Paragraph("Transkrip Mentah (STT)", cell_header_style),
        Paragraph("Teks Normalisasi", cell_header_style),
        Paragraph("Sentimen", cell_header_style),
        Paragraph("Skor", cell_header_style),
        Paragraph("Durasi", cell_header_style),
        Paragraph("Latensi", cell_header_style),
        Paragraph("Koreksi Manual", cell_header_style),
        Paragraph("Sentimen Koreksi", cell_header_style),
        Paragraph("Edit Dist", cell_header_style)
    ]

    table_data = [headers]

    for idx, r in enumerate(records, start=1):
        time_str = r.get("created_at_formatted") or "-"
        raw_text = r.get("text_raw") or "-"
        norm_text = r.get("text_normalized") or "-"
        sent = (r.get("sentiment") or "-").upper()
        score = f"{float(r.get('model_score')):.4f}" if r.get("model_score") is not None else "-"
        duration = f"{float(r.get('audio_duration')):.2f}s" if r.get("audio_duration") is not None else "-"
        exec_time = f"{float(r.get('execution_time')):.3f}s" if r.get("execution_time") is not None else "-"
        corr_text = r.get("user_corrected_text") or "-"
        corr_sent = (r.get("corrected_sentiment") or "-").upper() if r.get("corrected_sentiment") else "-"
        
        edit_dist = "-"
        if r.get("word_edit_distance") is not None:
            edit_dist = f"{r.get('word_edit_distance')}"

        # Color sentiment cell
        if "POSITIF" in sent:
            sent_html = f"<font color='#047857'><b>{sent}</b></font>"
        elif "NEGATIF" in sent:
            sent_html = f"<font color='#B91C1C'><b>{sent}</b></font>"
        elif "NETRAL" in sent:
            sent_html = f"<font color='#B45309'><b>{sent}</b></font>"
        else:
            sent_html = sent

        corr_sent_html = corr_sent
        if "POSITIF" in corr_sent:
            corr_sent_html = f"<font color='#047857'><b>{corr_sent}</b></font>"
        elif "NEGATIF" in corr_sent:
            corr_sent_html = f"<font color='#B91C1C'><b>{corr_sent}</b></font>"
        elif "NETRAL" in corr_sent:
            corr_sent_html = f"<font color='#B45309'><b>{corr_sent}</b></font>"

        row = [
            Paragraph(str(idx), cell_style_center),
            Paragraph(time_str, cell_style_center),
            Paragraph(raw_text, cell_style),
            Paragraph(norm_text, cell_style),
            Paragraph(sent_html, cell_style_center),
            Paragraph(score, cell_style_center),
            Paragraph(duration, cell_style_center),
            Paragraph(exec_time, cell_style_center),
            Paragraph(corr_text, cell_style),
            Paragraph(corr_sent_html, cell_style_center),
            Paragraph(edit_dist, cell_style_center)
        ]
        table_data.append(row)

    if len(records) == 0:
        empty_row = [Paragraph("<i>Belum ada riwayat rekaman tersimpan di database Neon.</i>", cell_style)] + [Paragraph("", cell_style)] * 10
        table_data.append(empty_row)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#312E81')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])

    # Alternating row colors
    for i in range(1, len(table_data)):
        bg = colors.HexColor('#F8FAFC') if i % 2 == 0 else colors.white
        table_style.add('BACKGROUND', (0, i), (-1, i), bg)

    t.setStyle(table_style)
    elements.append(t)

    doc.build(elements)
    buffer.seek(0)
    return buffer
