import pandas as pd
import glob
import re

def make_markdown(infile, outfile, num, name, sep=','):
  '''infile is a csv file, outfile is markdown'''
  print('\nprocessing', infile)
  df = pd.read_csv(infile, sep=sep)
  notebook_url = 'https://colab.research.google.com/github/probml/probml-notebooks/blob/master/notebooks'
  markdown_url = 'https://colab.research.google.com/github/probml/probml-notebooks/blob/master/markdown'
  d2l_url = 'https://colab.research.google.com/github/probml/probml-notebooks/blob/master/notebooks-d2l'

  contents = []
  contents.append(f'# Chapter {num} ({name}): Supplementary material')
  contents.append('|Title|Software|Link|')
  contents.append('-|-|-')

  for i in range(len(df)):
    entry = df.loc[i]
    title = entry['Title']
    lang = entry['Language']
    link = entry['Link']
    entry_type = entry['Type']
    if entry_type == 'Notebook':
      link = f'{notebook_url}/{link}'
    elif entry_type == 'Markdown':
      link = f'{markdown_url}/{link}'
    elif entry_type == 'd2lbook':
      link = f'{d2l_url}/{link}'
    line = f'|{title}|{lang}|[{entry_type}]({link})'
    contents.append(line)

  out = '\n'.join(contents)
  with open(outfile, 'w') as f:
    f.write(out)


def extract_chapter_num(fullname):
  # fullname is eg  '/Users/kpmurphy/github/pml-book/pml1/supplements/chap22.md'
  # return 22
  parts = fullname.split('/')
  fname = parts[-1]
  x = re.findall('\d{1,2}', fname)
  y=int(x[0])
  return y

def make_all_markdown(supplements_dir, titles):
  filenames = glob.glob(f"{supplements_dir}/*.csv")
  for f_csv in filenames:
      f_md = f_csv.replace('csv', 'md')
      num = extract_chapter_num(f_md)
      name = titles[num]
      print(f'making {f_md}, num {num}, name {name}')
      make_markdown(f_csv, f_md, num, name)


def get_chapter_titles(fname):
  df = pd.read_csv(fname, header=None, names=['Title'])
  dd = {}
  for i in range(len(df)):
    chapnum = i + 1
    chapname = df.loc[i]['Title']
    dd[chapnum] = chapname
  return dd


root='/Users/kpmurphy/github/pml-book/pml1'

titles = get_chapter_titles(f'{root}/TOC/chapters.txt')

make_all_markdown(f'{root}/supplements', titles)
