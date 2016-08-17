#data analysis on boxable-entities-v3.csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fname = '/Users/kpmurphy/github/pmtk3/data/boxable-entities-v3.csv'

# Prevent numpy from printing too many digits
np.set_printoptions(precision=3)

# Prevent Pandas from printing results in html format
pd.set_option('notebook_repr_html', False)
pd.set_option('display.max_columns', 160)
pd.set_option('display.width', 1000)


df = pd.read_csv(fname)
df = df.fillna(0)
bool_cols = ['ICAlocalizerV2', 'VOC2012', 'MSCOCO', 'MSCOCO2014', 
    'ImageNetDet2015', 'SUN3D', 'kpmurphy']
for c in bool_cols:
    df[c] = df[c].astype(bool) 

datasets = bool_cols
ndatasets = len(datasets)
counts = {'total':len(df['mid'])}
entities_in_dataset = {}
for dataset in datasets:
    mask = df[dataset] == True
    counts[dataset] = sum(mask)
    entities_in_dataset[dataset] = set((df['name'][mask]).tolist())
    
print counts

entities_with_data = set()
for d in ['VOC2012', 'MSCOCO2014', 'ImageNetDet2015', 'SUN3D']:
    entities_with_data = entities_with_data.union(entities_in_dataset[d])
print("{} entities have bounding box data".format(len(entities_with_data)))

entities_without_data = entities_in_dataset['ICAlocalizerV2'] - entities_with_data
print("{} entities without bounding box data".format(len(entities_without_data)))

print("num labels in VOC and ICA {}".format(sum(df['VOC2012'] & df['ICAlocalizerV2'])))
print("num labels in COCO and ICA {}".format(sum(df['MSCOCO2014'] & df['ICAlocalizerV2'])))

# Count how many instances of each category in each dataset
categories = np.unique(df['category'])
ncat = len(categories)
cat_counts = pd.DataFrame(columns=['category', 'count'] + bool_cols)
cat_counts['category'] = categories
for row, cat in enumerate(categories):
    cat_counts['count'][row] = sum(df['category']==cat)
    for dataset in datasets:
        in_dataset = df[dataset]==True
        has_cat = df['category']==cat
        cat_counts[dataset][row] = sum(in_dataset & has_cat)

print cat_counts
