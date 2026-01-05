from pypdf import PdfReader

pdf_path = "sample.pdf"  # put a PDF with this name in the same folder

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

text = ""
for i, page in enumerate(reader.pages):
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

print("\n--- Extracted Text (first 1000 characters) ---\n")
print(text[:1000])
