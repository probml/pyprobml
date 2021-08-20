from datetime import datetime
from github import Github
import pandas as pd
import os
pd.set_option("display.max_colwidth", 100)

public_access_token =  os.environ["GH_PUBLIC_ACCESS_TOKEN"]
g = Github(public_access_token)
repo = g.get_repo("probml/pyprobml")
print(repo)