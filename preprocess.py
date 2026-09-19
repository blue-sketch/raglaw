import fitz
import re

PDF_PATH = "cc.pdf"
OUTPUT_FILE = "clean.txt"


def clean_line(line):
    line = line.replace("\u00a0", " ")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


doc = fitz.open(PDF_PATH)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as out:

    for page_number, page in enumerate(doc, start=1):

        # Extract individual lines
        text = page.get_text("text")

        lines = text.splitlines()

        out.write(
            f"\n===== PAGE {page_number} =====\n\n"
        )

        for line in lines:

            line = clean_line(line)

            if line:
                out.write(line + "\n")


doc.close()

print("Done!")
print("Saved:", OUTPUT_FILE)