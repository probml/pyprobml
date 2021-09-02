# get github download stats for book
# Mahmoud Soliman <mjs@aucegypt.edu>


'''
a script to check downloads of a github release
make sure you have requests installed - pip3 install requests
usage python3 DownloadStats.py 
optional args:
-api_url url can be pointed to any other github repo to get the release stats
-hourly sleeps for an hour and checks again
'''
import superimport

import time
import requests
import json
import argparse, sys
from datetime import datetime
parser=argparse.ArgumentParser()
one_hour=60*60
one_day=24*one_hour
parser.add_argument('-api_url', help='github api url',default="https://api.github.com/repos/probml/pml-book/releases")
parser.add_argument('-hourly', help='check hourly',action='store_true')

args=parser.parse_args()

def check(api_url):
    response = requests.get(args.api_url)
    json_content=json.loads(response.content)
    header="release    | downloads | cumulative"
    line="--------------------------------"
    print(header)
    print(line)
    cumulative=0
    temp_d={}
    li=[]
    for element in json_content:
        for i in element["assets"]:
            temp_d["d"]=i["created_at"]
            temp_d["c"]=i["download_count"]
            temp_d["n"]=i["name"]
            li.append(temp_d)
            temp_d={}
        
    for i in reversed(li):
        cumulative=cumulative+i["c"]
        i["cumulative"]=cumulative      
    for i in li:
        print(i["d"][:10],"|"+str(i["c"]),"      |"+str(i["cumulative"]))
        
if args.hourly:
    while True:
        check(args.api_url)
        time.sleep(one_hour)
else:
    check(args.api_url)