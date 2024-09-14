from PyPDF2 import PdfMerger

merger = PdfMerger()

# 합칠 PDF 파일 경로 리스트
pdf_files = [
    "01시니어요가프로그램 활용 및 응용.pdf",
    "02시니어요가프로그램 활용 및 응용.pdf",
]

for pdf in pdf_files:
    merger.append(pdf)

merger.write("merged_file.pdf")
merger.close()
