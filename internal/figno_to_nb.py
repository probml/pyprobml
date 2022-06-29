from operator import index
import pandas as pd
book_no = 2
book_df = pd.read_csv(f'internal/figures_url_mapping_book{book_no}_excluded_dummy_nb.csv', dtype=str)
md_content = "## Book2\n"
for fig_no in book_df['key']:
    print(fig_no)
    md_content+=f"- [ ] {fig_no}\n"

with open(f"internal/figno_to_nb_book{book_no}.md", "w") as f:
    f.write(md_content)

# book_df["url"] = book_df["url"].apply(lambda x: list(map(lambda y: f"[{y.split('/')[-1]}]({y})", eval(x))))
# book_df["latex_added?"] = ["<li> - [ ] </li>"]*len(book_df)
# book_df.columns = ["Fig_no", "Notebook","Latex code added?"]
# book_df = book_df[["Fig_no","Latex code added?"]]
# book_df.to_markdown(f"internal/figno_to_nb_book{book_no}.md", index=False)