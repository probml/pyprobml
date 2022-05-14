import os
import re
import argparse
parser = argparse.ArgumentParser(description="delete firestore")
parser.add_argument("-user_name", "--user_name", type=str, help="")
args = parser.parse_args()

user_name = args.user_name
# Read number of jobs
with open(".github/workflows/notebooks.yml") as f:
    runners = re.findall("runner_id: (.*)", f.read())
    n_jobs = len(eval(runners[0]))

# Collect status, logs and figures
FOLDERS = ["workflow_testing_indicator", "auto_generated_figures"]
for folder in FOLDERS:
    if os.path.exists(folder):
        os.system("rm -rf " + folder)
    os.makedirs(folder)

repo = f"https://github.com/{user_name}/pyprobml.git"
for job in range(n_jobs):
    for folder in FOLDERS:
        branch = f"{folder}_{job}"
        if os.path.exists(branch):
            os.system(f"rm -rf {branch}")
        os.system(f"git clone --depth 1 --branch {branch} {repo} {branch}")
        os.system(f"cp -r {branch}/* {folder}")
        print(f"Copied {branch} to {folder}")
        os.system(f"rm -rf {branch}")
