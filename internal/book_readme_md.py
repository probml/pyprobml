import pandas as pd
import glob
import re

def get_chapter_titles(fname):
  df = pd.read_csv(fname, header=None, names=['Title'])
  dd = {}
  for i in range(len(df)):
    chapnum = i + 1
    chapname = df.loc[i]['Title']
    dd[chapnum] = chapname
  return dd


def make_readme(title_dict, outfile, include_colab_output=True):
    figure_url = 'https://colab.research.google.com/github/probml/pml-book/blob/master/pml1/figure_notebooks'
    supplements_url = 'https://github.com/probml/pml-book/blob/main/pml1/supplements'

    contents = []
    contents.append('# "Probabilistic Machine Learning: An Introduction"\n')
    if include_colab_output:
        contents.append('|Chapter|Name|Colab for generating figures|Colab with output|Supplementary material|')
        contents.append('-|-|-|-|-')
    else:
        contents.append('|Chapter|Name|Colab for figures|Supplementary material|')
        contents.append('-|-|-|-')

    for i in range(len(title_dict)):
        chap_num = i+1
        chap_name = title_dict[chap_num]
        simple_name = chap_name.lower()
        simple_name = simple_name.replace(':', '')
        simple_name = simple_name.replace(' ', '_')
        fig_url = f'{figure_url}/chapter{chap_num}_{simple_name}_figures.ipynb'
        fig_txt = f'[Link]({fig_url})'
        output_url = f'{figure_url}/chapter{chap_num}_{simple_name}_figures_output.ipynb'
        output_txt = f'[Link]({output_url})'
        supp_url = f'{supplements_url}/chap{chap_num}.md'
        supp_txt = f'[Link]({supp_url})'
        if include_colab_output:
            line = f'|{chap_num}|{chap_name}|{fig_txt}|{output_txt}|{supp_txt}|'
        else:
            line = f'|{chap_num}|{chap_name}|{fig_txt}|{supp_txt}|'
        contents.append(line)

    out = '\n'.join(contents)
    with open(outfile, 'w') as f:
        f.write(out)



root='/Users/kpmurphy/github/pml-book/pml1'

titles = get_chapter_titles(f'{root}/TOC/chapters.txt')

make_readme(titles, f'{root}/README2.md')