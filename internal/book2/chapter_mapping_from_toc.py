from TexSoup import TexSoup
import regex as re
from probml_utils.url_utils import dict_to_csv
import pandas as pd

lof_file_path = "internal/book2.toc"
with open(lof_file_path) as fp:
    LoF_File_Contents = fp.read()
soup = TexSoup(LoF_File_Contents)

chap_no_pattern = "numberline{(\d*)?}"
chap_name_pattern = "numberline{\d*?}(.*?})"
chap_no_to_name = {}

for each in soup.find_all("contentsline")[2:-1]:
    if "contentsline{chapter}" in str(each):
        # print(each)
        chap_no = re.findall(chap_no_pattern, str(each))[0]
        chap_name = re.findall(chap_name_pattern, str(each))[0][:-1]
        chap_no_to_name[chap_no] = chap_name
print(chap_no_to_name)
dict_to_csv(chap_no_to_name, "internal/chapter_no_to_name_mapping_book2.csv", columns=["chap_no", "chap_name"])

df = pd.read_csv("internal/chapter_no_to_name_mapping_book2.csv")
df["Notebook"] = df["chap_no"].apply(lambda x: f"[{x:02d}/]({x:02d}/)")

md = "# Probabilistic Machine Learning: Advanced Topics\n\n## Chapters\n\n" + df.to_markdown(index=False)
with open("notebooks/book2/README.md", "w") as fp:
    fp.write(md)
