import os
import argparse
import nbformat as nbf
from probml_utils.nb_utils import get_ipynb_from_code

parser = argparse.ArgumentParser(description="Handle book2 pycode")
parser.add_argument("-bookgen", "--bookgen", type=int, choices=[0, 1], default=0, help="")
args = parser.parse_args()

# configs
macro = r"\newcommand{\pycode}[1]{\href{https://github.com/probml/pyprobml/blob/master/scripts/#1.py}{#1.py}}"
replaced_macro = r"\newcommand{\pycode}[1]{\href{https://github.com/probml/pyprobml/blob/master/scripts/#1.py}{ssssspy\twodigits{\thechapter}/#1.pyeeeeepy}}"

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
        f"cd {book_root}/{bookname}; {pdflatex_cmd}; {bibtex_cmd}; {pdflatex_cmd}; mv {bookname}.pdf {bookname}_pycode.pdf"
    )

    # restore the macro
    with open(os.path.join(book_root, "macros.tex"), "r") as f:
        content = f.read()
    with open(os.path.join(book_root, "macros.tex"), "w") as f:
        f.write(content.replace(replaced_macro, macro))

# read the book
import fitz

with fitz.open(os.path.join(book_root, bookname, f"{bookname}_pycode.pdf")) as doc:
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

all_scripts = set(re.findall("ssssspy(.*?)eeeeepy", text_one_line))
# print(all_scripts)

# start transfer
for chap_script in all_scripts:
    chap, script = chap_script.split("/")
    try:
        with open("scripts/" + script, "r") as f:
            nb = get_ipynb_from_code(f.read())
        save_path = f"notebooks/book2/{chap}/{script.replace('.py', '.ipynb')}"
        save_dir = os.path.dirname(save_path)
        assert not os.path.exists(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, "w") as f:
            nbf.write(nb, f)
        # print("done", chap, script)
    except Exception as e:
        print(e, chap, script)
