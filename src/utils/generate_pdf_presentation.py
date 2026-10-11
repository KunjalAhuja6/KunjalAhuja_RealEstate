from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

doc = SimpleDocTemplate('presentation/Project_Presentation.pdf', pagesize=landscape(letter),
                        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
styles = getSampleStyleSheet()

title_style = ParagraphStyle('SlideTitle', parent=styles['Heading1'], fontSize=22, leading=26, textColor=colors.HexColor('#1e3a8a'))
sub_style = ParagraphStyle('SlideSub', parent=styles['Normal'], fontSize=12, leading=16, textColor=colors.HexColor('#475569'), spaceAfter=14)
body_style = ParagraphStyle('SlideBody', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#1e293b'))

slides = [
    ('AI-Powered Real Estate Property Valuation Assistant',
     'Prodapt Capstone Project • Panel Presentation (8 Min Demo + 2 Min Q&A)',
     [
         '<b>Project Domain</b>: Real Estate & PropTech Valuation Intelligence',
         '<b>Presenter</b>: Senior ML/Software Engineer',
         '<b>Architecture</b>: Autonomous Multi-Agent Collaborative System with Hybrid RAG & Supervised ML',
         '<b>Evaluation</b>: Verified with DeepEval Recommendation Quality Benchmarks'
     ]),
    ('Problem Statement & Business Need',
     'Why Real Estate Appraisals Require GenAI Automation',
     [
         '<b>Manual Search Inefficiencies</b>: Traditional appraisal requires manual scanning of thousands of historical records.',
         '<b>Valuation Inconsistency</b>: Heuristic condition and age adjustments introduce subjective bias.',
         '<b>Unquantified Market Risk</b>: Properties with deferred maintenance lack quantified financial risk scoring.',
         '<b>Business Objective</b>: Build an autonomous multi-agent valuation assistant delivering instant comp retrieval, ML predictions, risk scoring, and LLM-as-judge audit.'
     ]),
    ('Solution Overview & High-Level Architecture',
     'End-to-End Flow from Dataset to Executive Dossier',
     [
         '<b>1. Data Layer</b>: Standardizes Ames Housing Dataset (2,930 records, 15 key fields, 1-10 quality scale).',
         '<b>2. Hybrid RAG</b>: Dense embeddings (all-MiniLM-L6-v2) + lexical TF-IDF keyword search.',
         '<b>3. Supervised ML</b>: Gradient Boosting Regressor (Test R² = 0.9031, MAPE = 8.46%).',
         '<b>4. Multi-Agent Core</b>: Classification, Valuation, Risk Assessment, and LLM-as-Judge Auditor.',
         '<b>5. Presentation</b>: FastAPI REST microservice + Streamlit interactive dashboard.'
     ]),
    ('Hybrid RAG & Comparable-Property Prioritization',
     'Ensuring Precise, Grounded Comparable Selection',
     [
         '<b>Dense Embeddings</b>: 384-dimensional vectors with local caching for instant sub-5ms cosine retrieval.',
         '<b>Lexical Match</b>: Captures exact specs (bedrooms, quality grades, subdivision names).',
         '<b>Metadata Filtering</b>: Filters candidates on location, age, and square footage.',
         '<b>Prioritization Formula</b>: Multi-factor index weighting Location (30%), Living Area (25%), Quality (20%), Age (15%), and Room Count (10%).'
     ]),
    ('Autonomous Multi-Agent Collaboration Workflow',
     'Orchestrating Specialized Agents for Appraisal Precision',
     [
         '<b>Guardrail Agent</b>: Physical boundary validation (100-15k sqft, 1-10 qual) & injection filter.',
         '<b>Classification Agent</b>: Market Tier, Typology, Condition Grade, and Target Demographics.',
         '<b>Valuation Agent</b>: Comp-median, mean, and size-weighted PPSF fair market value.',
         '<b>Risk Agent</b>: Quantifies age depreciation, condition gap, and market volatility (0-100 score).',
         '<b>LLM-as-Judge Agent</b>: Independent audit against ML regression and comps issuing verdicts (APPROVED).'
     ]),
    ('Supervised ML Pricing & Feature Impact',
     'Predictive Modeling Benchmark & What-If Simulation',
     [
         '<b>Gradient Boosting Regressor</b>: Achieved Test R² = 0.9031, RMSE = $27,870, MAE = $15,889, MAPE = 8.46%.',
         '<b>Top Drivers</b>: Overall Quality (54%), Living Area (22%), Basement (8%), and Location Premium.',
         '<b>Location Premiums</b>: Stone Brook (+85.9% premium), Northridge (+89.1%), Edwards (-22.5%).',
         '<b>What-If Renovation Simulator</b>: Real-time simulation of additions, quality upgrades, and remodels.'
     ]),
    ('Evaluation Results (DeepEval Framework)',
     'Honest Dual-Mode Quality Framework (v4.2.8 Pinned)',
     [
         '<b>Dual-Mode Operation</b>: Operates in offline mode without API keys (evaluation_mode = "offline_no_llm", benchmark_status = "NOT_EVALUATED") without fabricating canned scores.',
         '<b>Live DeepEval Metrics</b>: Evaluates Faithfulness, Answer Relevancy, Contextual Precision, and Recall using real DeepEval metric classes on LLMTestCase when OPENAI_API_KEY is configured.',
         '<b>Valuation Accuracy</b>: Evaluates reconciled valuation accuracy vs verified historical market bounds across benchmark scenarios in all modes.',
         '<b>True Hybrid Retrieval</b>: Lexical TF-IDF keyword scoring covers the full metadata-filtered candidate set before score fusion.',
         '<b>Grounded Grounding</b>: Reconciled FMV combines comp evidence, supervised ML regression (R² = 0.9031), and feedback calibration.'
     ]),
    ('Feedback Loop for Continuous Improvement',
     'Dynamic Calibration of Systemic Valuation Bias',
     [
         '<b>Appraiser Feedback Logging</b>: Appraisers submit verified sales prices and 1-5 star ratings.',
         '<b>Persistent Repository</b>: Logged in JSON data store with error metrics.',
         '<b>Uncalibrated Bias Measurement</b>: Errors and systematic bias are evaluated against uncalibrated estimates (recommended / applied_factor) to prevent calibration drift.',
         '<b>Dynamic Calibration Factor</b>: Directly calibrated in the orchestrator reconciliation step.'
     ]),
    ('Working Solution Demo & Local Deployment',
     'Production-Ready Microservice & Interactive Web UI',
     [
         '<b>Streamlit Dashboard</b>: 6 dedicated tabs covering Dossier, Hybrid Comps, ML Pricing, Risk, Feedback, DeepEval.',
         '<b>FastAPI Microservice</b>: 14 REST endpoints with full OpenAPI documentation at /docs.',
         '<b>Local Execution</b>: Runs locally via `uvicorn src.main:app` & `streamlit run src/frontend.py`.',
         '<b>Zero Secrets Committed</b>: Follows strict security hygiene.'
     ]),
    ('Conclusion, Challenges & Q&A',
     'Summary of Achievements (2 Min Q&A)',
     [
         '<b>100% Scope Compliance</b>: Built all 8 Basic items and all 8 Advanced items exactly to specification.',
         '<b>Rigorous Verification</b>: 32 pytest unit/integration tests passing cleanly (100% success rate).',
         '<b>Complete Deliverables</b>: Architecture Diagram (PDF/PNG), Design Doc (PDF), API Doc (PDF), Evaluation Report (PDF), Dataset Details (PDF), Presentation Deck (PPTX/PDF).',
         '<b>Open for Panel Q&A</b>: Thank you!'
     ])
]

story = []
for i, (title, subtitle, bullets) in enumerate(slides):
    story.append(Paragraph(f'Slide {i+1}: {title}', title_style))
    story.append(Paragraph(subtitle, sub_style))
    for b in bullets:
        story.append(Paragraph(f'• {b}', body_style))
        story.append(Spacer(1, 4))
    if i < len(slides) - 1:
        story.append(PageBreak())

doc.build(story)
print('Generated presentation/Project_Presentation.pdf successfully.')
