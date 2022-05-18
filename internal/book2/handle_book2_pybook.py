import os
import argparse
import shutil

parser = argparse.ArgumentParser(description="Handle book2 pybook")
parser.add_argument("-bookgen", "--bookgen", type=int, choices=[0, 1], default=0, help="")
args = parser.parse_args()

# configs
macro = r"\newcommand{\pybook}[1]{\href{https://colab.research.google.com/github/probml/probml-notebooks/blob/master/notebooks/#1.ipynb}{#1.ipynb}}"
replaced_macro = r"\newcommand{\pybook}[1]{\href{https://colab.research.google.com/github/probml/probml-notebooks/blob/master/notebooks/#1.ipynb}{sssssnb\twodigits{\thechapter}/#1.ipynbeeeeenb}}"

book_root = "../bookv2"
bookname = "book2"

if args.bookgen:
    # replace the macro
    with open(os.path.join(book_root, "macros.tex"), "r") as f:
        content = f.read()
    with open(os.path.join(book_root, "macros.tex"), "w") as f:
        f.write(content.replace(macro, replaced_macro))

    # generate the book
    pdflatex_cmd = f"pdflatex --interaction=nonstopmode {bookname}"
    bibtex_cmd = f"bibtex {bookname}"
    os.system(
        f"cd {book_root}/{bookname}; {pdflatex_cmd}; {bibtex_cmd}; {pdflatex_cmd}; mv {bookname}.pdf {bookname}_pybook.pdf"
    )

    # restore the macro
    with open(os.path.join(book_root, "macros.tex"), "r") as f:
        content = f.read()
    with open(os.path.join(book_root, "macros.tex"), "w") as f:
        f.write(content.replace(replaced_macro, macro))

# read the book
import fitz

with fitz.open(os.path.join(book_root, bookname, f"{bookname}_pybook.pdf")) as doc:
    text = ""
    for page in doc:
        text += page.get_text() + ""
text_one_line = text.replace("\n", "")

# solve some glitches
glitch_dict = {"ﬁ": "fi", "ﬀ": "ff", "ﬂ": "fl"}
for glitch, fix in glitch_dict.items():
    text_one_line = text_one_line.replace(glitch, fix)

# parse script names
import re

all_nb = set(re.findall("sssssnb(.*?)eeeeenb", text_one_line))
# print(all_nb)

# start transfer
for chap_nb in all_nb:
    chap, nb = chap_nb.split("/")
    try:
        save_path = f"notebooks/book2/{chap}/{nb}"
        try:
            assert not os.path.exists(save_path)
        except AssertionError:
            raise Exception(f"{save_path} already exists")
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copy(f"../probml-notebooks/notebooks/{nb}", save_path)
        # print("done", chap, script)
    except Exception as e:
        print(e)
