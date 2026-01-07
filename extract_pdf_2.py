import sys
try:
    import PyPDF2
    reader = PyPDF2.PdfReader(sys.argv[1])
    for page in reader.pages:
        print(page.extract_text())
except ImportError:
    print("PyPDF2 not found")
