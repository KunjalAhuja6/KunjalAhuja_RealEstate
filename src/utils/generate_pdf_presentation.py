"""
Legacy PDF Presentation Wrapper
Delegates to generate_presentation.py to ensure consistency across all runs.
Author / Presenter: Kunjal Ahuja
"""

import subprocess
import sys
import os

if __name__ == '__main__':
    from src.utils.generate_presentation import generate_pdf, generate_pptx
    pdf_path = 'presentation/Project_Presentation.pdf'
    pptx_path = 'presentation/Project_Presentation.pptx'
    generate_pptx(pptx_path)
    generate_pdf(pdf_path)
    print("Project presentation generation complete.")
