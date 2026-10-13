"""
Architecture Diagram Generator
Renders Architecture_Diagram.png (high-res 2400x1600 raster)
and Architecture_Diagram.pdf (pure vector PDF)
Conforms strictly to the required pipeline stages and labels:
Dataset -> Preprocessing -> Chunking -> Embeddings -> Vector Database -> Retriever -> LLM/RAG -> Agents -> API Layer -> User Interface
Shows labeled synthetic data step and real model (GradientBoostingRegressor, R2 = 0.9198).
Author / Presenter: Kunjal Ahuja
"""

import os
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors


def generate_png_diagram(png_path):
    width = 2400
    height = 1600
    img = Image.new("RGBA", (width, height), (248, 250, 252, 255))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 44)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        stage_title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        body_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        tag_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        title_font = subtitle_font = stage_title_font = body_font = tag_font = ImageFont.load_default()

    # Header Banner
    draw.rectangle([0, 0, width, 120], fill=(30, 58, 138, 255))
    draw.text((60, 24), "AI-Powered Real Estate Property Valuation Assistant", fill=(255, 255, 255, 255), font=title_font)
    draw.text((60, 78), "End-to-End System Architecture • Presenter: Kunjal Ahuja • Prodapt Capstone Project", fill=(191, 219, 254, 255), font=subtitle_font)

    def draw_card(box, title, stage_num, stage_label, bullet_points, header_bg, card_bg=(255, 255, 255, 255)):
        x0, y0, x1, y1 = box
        draw.rounded_rectangle([x0+4, y0+4, x1+4, y1+4], radius=12, fill=(226, 232, 240, 180))
        draw.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=card_bg, outline=(203, 213, 225, 255), width=2)
        draw.rounded_rectangle([x0, y0, x1, y0+48], radius=12, fill=header_bg)
        draw.rectangle([x0, y0+36, x1, y0+48], fill=header_bg)
        draw.text((x0+16, y0+12), f"{stage_num}. {stage_label.upper()}", fill=(255, 255, 255, 255), font=stage_title_font)
        draw.text((x0+16, y0+58), title, fill=(30, 41, 59, 255), font=stage_title_font)
        curr_y = y0 + 94
        for bp in bullet_points:
            draw.ellipse([x0+18, curr_y+6, x0+24, curr_y+12], fill=(13, 148, 136, 255))
            draw.text((x0+32, curr_y), bp, fill=(51, 65, 85, 255), font=body_font)
            curr_y += 28

    def draw_arrow(start, end):
        sx, sy = start
        ex, ey = end
        draw.line([sx, sy, ex, ey], fill=(100, 116, 139, 255), width=4)
        if ex > sx:
            draw.polygon([(ex, ey), (ex-12, ey-8), (ex-12, ey+8)], fill=(100, 116, 139, 255))
        elif ex < sx:
            draw.polygon([(ex, ey), (ex+12, ey-8), (ex+12, ey+8)], fill=(100, 116, 139, 255))
        elif ey > sy:
            draw.polygon([(ex, ey), (ex-8, ey-12), (ex+8, ey-12)], fill=(100, 116, 139, 255))
        elif ey < sy:
            draw.polygon([(ex, ey), (ex-8, ey+12), (ex+8, ey+12)], fill=(100, 116, 139, 255))

    # Row 1: Stages 1-5
    draw_card([60, 160, 480, 560], "Ames Housing Dataset", "1", "Dataset",
              ["Total: 10,000 Records", "• 2,930 Real Transactions", "  (OpenML 43926 / De Cock)", "• 7,070 Synthetic Records", "  (Gaussian Copula + Novelty)", "28 Distinct Neighborhoods", "Year Built: 1872 – 2010", "15 Key Physical Fields"],
              header_bg=(30, 58, 138, 255))

    draw_card([540, 160, 960, 560], "Input Guardrails & ETL", "2", "Preprocessing",
              ["15 Key Fields Standardized", "1-10 Ordinal Quality Scale", "Derived Features:", "  Age, EffAge, PricePerSqFt", "Input Guardrail Validation:", "  14 Fields Checked", "  100 <= SqFt <= 15,000", "  Prompt Injection Defense"],
              header_bg=(30, 58, 138, 255))

    draw_card([1020, 160, 1440, 560], "Document Serialization", "3", "Chunking",
              ["1 Property = 1 Document", "No Splitting / No Overlap", "Searchable Text Payload:", "  Location & Subdivision", "  Living & Lot Area SqFt", "  Beds, Baths, Total Rooms", "  Year Built & Remodeled", "  Price & Historical PPSF"],
              header_bg=(30, 58, 138, 255))

    draw_card([1500, 160, 1920, 560], "Dense Representation", "4", "Embeddings",
              ["SentenceTransformers", "Model: all-MiniLM-L6-v2", "Dimension: 384-d Dense", "Local Offline Inference", "Precomputed & Cached", "Unit Normalized Vectors", "Fast Cosine Dot Product", "Zero Cloud Latency"],
              header_bg=(30, 58, 138, 255))

    draw_card([1980, 160, 2340, 560], "Vector Storage Layer", "5", "Vector Database",
              ["In-Memory NumPy Index", "embeddings.npy Matrix", "Shape: (10,000 x 384)", "Local Memory: ~15.3 MB", "Sub-5ms Cosine Query", "Zero DB Server Overhead", "Full Candidate Indexing", "Metadata Filter Cache"],
              header_bg=(30, 58, 138, 255))

    draw_arrow((480, 360), (540, 360))
    draw_arrow((960, 360), (1020, 360))
    draw_arrow((1440, 360), (1500, 360))
    draw_arrow((1920, 360), (1980, 360))
    draw_arrow((2160, 560), (2160, 620))
    draw_arrow((2160, 620), (240, 620))
    draw_arrow((240, 620), (240, 660))

    # Row 2: Stages 6-8
    draw_card([60, 660, 580, 1100], "True Hybrid Search Engine", "6", "Retriever",
              ["Full-Candidate TF-IDF Matrix", "Dense Cosine Semantic Fusion", "Formula: 0.55*Dense + 0.45*TFIDF", "Comparability Prioritization:", "  Location (30%), SqFt (25%)", "  Quality (20%), Age (15%), Rooms", "Real Comps Filter (No Synthetic)", "Dynamic Neighborhood Widening:", "  Expands Citywide if < 3 Comps"],
              header_bg=(15, 118, 110, 255))

    draw_card([640, 660, 1160, 1100], "Grounded Retrieval Synthesis", "7", "LLM / RAG",
              ["Dual-Mode Generation:", "  Live: OpenAI gpt-4o-mini", "  Offline: Heuristic Engine", "Grounded Valuation Narrative:", "  Cites Verified Comps & PPSF", "  Zero Financial Hallucinations", "Context Injection via Pydantic", "DeepEval Test Case Formulation", "Dynamic Parameterization"],
              header_bg=(15, 118, 110, 255))

    draw_card([1220, 660, 2340, 1100], "Multi-Agent Collaborative System Core", "8", "Agents",
              ["Autonomous Collaboration Orchestrator:",
               "• Classification Agent: Market Tier, Typology, Condition Grade, Demographics",
               "• Supervised ML Regressor: GradientBoostingRegressor (Test R² = 0.9198, RMSE = $25,361.84, MAPE = 8.87%)",
               "  Honest Real-Only Test Evaluation (n=586, random_state=42) + 90% Empirical Prediction Intervals",
               "• Basic Valuation Agent: Median Comparable Price ($) + Size-Weighted PPSF Valuation",
               "• Dynamic Reconciliation: comp_fit weighting (comp_weight: 0.1–0.5, ml_weight: 0.5–0.9)",
               "• Property Risk Agent: 0-100 Risk Score (Physical Age, Condition Gap, Volatility, Liquidity)",
               "• Feature Impact Agent: Neighborhood Location Premiums & What-If Renovation Simulator",
               "• LLM-as-Judge Auditor: Multi-criterion valuation audit issuing formal verdict (APPROVED)"],
              header_bg=(15, 118, 110, 255))

    draw_arrow((580, 880), (640, 880))
    draw_arrow((1160, 880), (1220, 880))
    draw_arrow((1780, 1100), (1780, 1160))
    draw_arrow((1780, 1160), (320, 1160))
    draw_arrow((320, 1160), (320, 1200))

    # Row 3: Stages 9-10 + Feedback
    draw_card([60, 1200, 780, 1550], "FastAPI REST Microservice Layer", "9", "API Layer",
              ["16 REST Endpoints (Base /api/v1):",
               "  GET /, GET /health, POST /guardrails/validate",
               "  POST /search, POST /classify, POST /valuation/basic",
               "  POST /valuation/predict-price, POST /valuation/predict-risk",
               "  POST /valuation/impact-analysis, POST /valuation/collaborate",
               "  POST /valuation/judge-validate, POST /feedback/submit",
               "  GET /feedback/summary, GET /evaluation/run, GET /evaluation/history",
               "  GET /dataset/info • OpenAPI Swagger UI at /docs"],
              header_bg=(30, 41, 59, 255))

    draw_card([840, 1200, 1600, 1550], "React 18 + Vite Production Frontend", "10", "User Interface",
              ["Modern SPA Architecture (frontend/):",
               "• Left Sidebar: Shared 14-Field Property Input Form",
               "• 7 Dedicated Accessible Workflow Tabs:",
               "  1. Valuation (RangeBar, 2-Sentence Plain-English Rationale)",
               "  2. Comparable Sales (Verified Historical Transactions)",
               "  3. ML Prediction (Gradient Boosting Point & 90% Interval)",
               "  4. What-If Simulator (Interactive Renovation Value Lift)",
               "  5. Risk & Audit (0-100 Multi-Factor Risk & Judge Verdict)",
               "  6. Feedback (Appraiser Rating & Bias Calibration)",
               "  7. Evaluation (DeepEval Quality Benchmark Runner)",
               "Theme: Dark/Light Mode with System Default & LocalStorage"],
              header_bg=(30, 41, 59, 255))

    draw_card([1660, 1200, 2340, 1550], "Continuous Learning & DeepEval Benchmark", "★", "Evaluation & Feedback",
              ["DeepEval v4.2.8 Quality Suite (5 Scenarios):",
               "• 100% Pass Rate (5/5 Passed in 61.98s live run)",
               "• Faithfulness: 0.900 | Answer Relevancy: 0.880",
               "• Context Precision: 0.850 | Context Recall: 0.850",
               "• Valuation Accuracy: 1.000",
               "Continuous Improvement Feedback Loop:",
               "• Logs Appraiser Ratings & Verified Closing Prices",
               "• Uncalibrated Bias Tracking (Prevents Drift)",
               "• Dynamic Calibration Factor Applied in Real-Time"],
              header_bg=(109, 40, 217, 255))

    draw_arrow((780, 1375), (840, 1375))
    draw_arrow((1600, 1375), (1660, 1375))

    img.save(png_path, "PNG", dpi=(300, 300))
    print(f"Successfully rendered {png_path} ({width}x{height})")


def generate_vector_pdf_diagram(pdf_path):
    # Vector PDF rendering using ReportLab Landscape Letter
    c = canvas.Canvas(pdf_path, pagesize=landscape(letter))
    pw, ph = landscape(letter) # 792 x 612
    margin_x = 24
    margin_y = 20
    draw_w = pw - (margin_x * 2) # 744
    draw_h = ph - (margin_y * 2) # 572
    
    # Coordinate transformation: map (0..2400, 0..1600) to (margin_x..margin_x+draw_w, ph-margin_y..margin_y)
    def tx(x):
        return margin_x + (x / 2400.0) * draw_w
    def ty(y):
        return (ph - margin_y) - (y / 1600.0) * draw_h
    def tw(w):
        return (w / 2400.0) * draw_w
    def th(h):
        return (h / 1600.0) * draw_h

    # Background
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.rect(0, 0, pw, ph, fill=1, stroke=0)

    # Header Banner
    banner_h = th(110)
    c.setFillColor(colors.HexColor("#1e3a8a"))
    c.rect(margin_x, ty(110), draw_w, banner_h, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin_x + 16, ty(40), "AI-Powered Real Estate Property Valuation Assistant")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(colors.HexColor("#bfdbfe"))
    c.drawString(margin_x + 16, ty(80), "End-to-End System Architecture • Author / Presenter: Kunjal Ahuja • Prodapt Capstone Project")

    def draw_vcard(box, title, stage_num, stage_label, bullets, header_hex):
        x0, y0, x1, y1 = box
        vx0, vy1, vw, vh = tx(x0), ty(y1), tw(x1 - x0), th(y1 - y0)
        # Background
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.setLineWidth(0.8)
        c.roundRect(vx0, vy1, vw, vh, 4, fill=1, stroke=1)
        # Header bar
        header_h = th(42)
        c.setFillColor(colors.HexColor(header_hex))
        c.rect(vx0, vy1 + vh - header_h, vw, header_h, fill=1, stroke=0)
        # Header text
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(vx0 + 6, vy1 + vh - th(26), f"{stage_num}. {stage_label.upper()}")
        # Subtitle
        c.setFillColor(colors.HexColor("#1e293b"))
        c.setFont("Helvetica-Bold", 7.0)
        c.drawString(vx0 + 6, vy1 + vh - th(62), title)
        # Bullets
        c.setFont("Helvetica", 5.8)
        c.setFillColor(colors.HexColor("#334155"))
        b_y = vy1 + vh - th(86)
        for b in bullets:
            c.drawString(vx0 + 8, b_y, b)
            b_y -= th(24)

    def draw_varrow(start, end):
        sx, sy = tx(start[0]), ty(start[1])
        ex, ey = tx(end[0]), ty(end[1])
        c.setStrokeColor(colors.HexColor("#64748b"))
        c.setLineWidth(1.2)
        c.line(sx, sy, ex, ey)

    # Row 1: Stages 1-5
    draw_vcard([60, 150, 480, 530], "Ames Housing Dataset", "1", "Dataset",
               ["Total: 10,000 Records", "• 2,930 Real Sales (OpenML 43926)", "• 7,070 Synthetic Records (Copula)", "28 Neighborhoods", "Year Built: 1872 – 2010", "15 Key Physical Fields"],
               "#1e3a8a")

    draw_vcard([540, 150, 960, 530], "Input Guardrails & ETL", "2", "Preprocessing",
               ["15 Key Fields Standardized", "1-10 Ordinal Quality Scale", "Derived: Age, EffAge, PPSF", "Guardrails: 14 Fields Validated", "100 <= SqFt <= 15,000", "Prompt Injection Defense"],
               "#1e3a8a")

    draw_vcard([1020, 150, 1440, 530], "Document Serialization", "3", "Chunking",
               ["1 Property = 1 Document", "No Splitting / No Overlap", "Searchable Text Payload:", "Location, SqFt, Beds, Baths", "Year Built & Remodeled", "Price & Historical PPSF"],
               "#1e3a8a")

    draw_vcard([1500, 150, 1920, 530], "Dense Representation", "4", "Embeddings",
               ["SentenceTransformers", "Model: all-MiniLM-L6-v2", "Dimension: 384-d Dense", "Local Offline Inference", "Cached Unit Vectors (.npy)", "Sub-millisecond Query"],
               "#1e3a8a")

    draw_vcard([1980, 150, 2340, 530], "Vector Storage Layer", "5", "Vector Database",
               ["In-Memory NumPy Index", "embeddings.npy Matrix", "Shape: (10,000 x 384)", "Size: ~15.3 MB in Memory", "Sub-5ms Cosine Similarity", "Zero Database Overhead"],
               "#1e3a8a")

    draw_varrow((480, 340), (540, 340))
    draw_varrow((960, 340), (1020, 340))
    draw_varrow((1440, 340), (1500, 340))
    draw_varrow((1920, 340), (1980, 340))

    # Row 2: Stages 6-8
    draw_vcard([60, 590, 580, 1030], "True Hybrid Search Engine", "6", "Retriever",
               ["Candidate TF-IDF Matrix", "Dense Cosine Similarity", "Fusion: 0.55*Dense + 0.45*TFIDF", "Comparability Index:", "Location 30%, SqFt 25%, Qual 20%", "Age 15%, Room Count 10%", "Comps Filter: Exclude Synthetic", "Dynamic Neighborhood Widening"],
               "#0f766e")

    draw_vcard([640, 590, 1160, 1030], "Grounded Retrieval Synthesis", "7", "LLM / RAG",
               ["Dual-Mode Operation:", "• Live: OpenAI gpt-4o-mini", "• Offline: Heuristic Engine", "Grounded Valuation Narrative", "Cites Verified Comps & PPSF", "Zero Financial Hallucination", "Pydantic Schema Validation", "DeepEval Benchmark Integration"],
               "#0f766e")

    draw_vcard([1220, 590, 2340, 1030], "Multi-Agent Collaborative System Core", "8", "Agents",
               ["Collaborative Multi-Agent Architecture:",
                "• Classification Agent: Market Tier, Typology, Condition Grade, Demographics",
                "• Supervised Regressor: GradientBoostingRegressor (Test R² = 0.9198, RMSE = $25,361.84, MAPE = 8.87%)",
                "  Honest Real-Only Test Set (n=586, random_state=42) + 90% Empirical Prediction Intervals",
                "• Basic Valuation Agent: Median Comparable Price ($) + Size-Weighted PPSF Valuation",
                "• Dynamic Reconciliation: comp_fit weighting (comp_weight: 0.1–0.5, ml_weight: 0.5–0.9)",
                "• Property Risk Agent: 0-100 Risk Score (Physical Age, Condition Gap, Volatility, Liquidity)",
                "• Feature Impact Agent: Neighborhood Location Premiums & What-If Renovation Simulator",
                "• LLM-as-Judge Auditor: Multi-criterion valuation audit issuing formal verdict (APPROVED)"],
               "#0f766e")

    draw_varrow((580, 810), (640, 810))
    draw_varrow((1160, 810), (1220, 810))

    # Row 3: Stages 9-10 + Feedback
    draw_vcard([60, 1090, 780, 1530], "FastAPI REST Microservice Layer", "9", "API Layer",
               ["16 REST Endpoints (Base /api/v1):",
                "  GET /, GET /health, POST /guardrails/validate",
                "  POST /search, POST /classify, POST /valuation/basic",
                "  POST /valuation/predict-price, POST /valuation/predict-risk",
                "  POST /valuation/impact-analysis, POST /valuation/collaborate",
                "  POST /valuation/judge-validate, POST /feedback/submit",
                "  GET /feedback/summary, GET /evaluation/run, GET /evaluation/history",
                "  GET /dataset/info • OpenAPI Swagger UI at /docs"],
               "#1e293b")

    draw_vcard([840, 1090, 1600, 1530], "React 18 + Vite Production Frontend", "10", "User Interface",
               ["Modern SPA Architecture (frontend/):",
                "• Left Sidebar: Shared 14-Field Property Input Form",
                "• 7 Dedicated Accessible Workflow Tabs:",
                "  1. Valuation (RangeBar, 2-Sentence Plain-English Rationale)",
                "  2. Comparable Sales (Verified Historical Transactions)",
                "  3. ML Prediction (Gradient Boosting Point & 90% Interval)",
                "  4. What-If Simulator (Interactive Renovation Value Lift)",
                "  5. Risk & Audit (0-100 Multi-Factor Risk & Judge Verdict)",
                "  6. Feedback (Appraiser Rating & Bias Calibration)",
                "  7. Evaluation (DeepEval Quality Benchmark Runner)",
                "Theme: Dark/Light Mode with System Default & LocalStorage"],
               "#1e293b")

    draw_vcard([1660, 1090, 2340, 1530], "Continuous Learning & DeepEval Benchmark", "★", "Evaluation & Feedback",
               ["DeepEval v4.2.8 Quality Suite (5 Scenarios):",
                "• 100% Pass Rate (5/5 Passed in 61.98s live run)",
                "• Faithfulness: 0.900 | Answer Relevancy: 0.880",
                "• Context Precision: 0.850 | Context Recall: 0.850",
                "• Valuation Accuracy: 1.000",
                "Continuous Improvement Feedback Loop:",
                "• Logs Appraiser Ratings & Verified Closing Prices",
                "• Uncalibrated Bias Tracking (Prevents Drift)",
                "• Dynamic Calibration Factor Applied in Real-Time"],
               "#6d28d9")

    draw_varrow((780, 1310), (840, 1310))
    draw_varrow((1600, 1310), (1660, 1310))

    c.save()
    print(f"Successfully generated vector {pdf_path} (size: {os.path.getsize(pdf_path)} bytes)")


if __name__ == '__main__':
    png = 'architecture/Architecture_Diagram.png'
    pdf = 'architecture/Architecture_Diagram.pdf'
    generate_png_diagram(png)
    generate_vector_pdf_diagram(pdf)
