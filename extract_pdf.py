import sys
try:
    from pypdf import PdfReader
    reader = PdfReader(sys.argv[1])
    for page in reader.pages:
        print(page.extract_text())
except ImportError:
    print("pypdf not found")
