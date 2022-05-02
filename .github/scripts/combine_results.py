import os
import re

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

repo = "https://github.com/patel-zeel/pyprobml.git"
for job in range(n_jobs):
    for folder in FOLDERS:
        branch = f"{folder}_{job}"
        if os.path.exists(branch):
            os.system(f"rm -rf {branch}")
        os.system(f"git clone --depth 1 --branch {branch} {repo} {branch}")
        os.system(f"cp -r {branch}/* {folder}")
        print(f"Copied {branch} to {folder}")
        os.system(f"rm -rf {branch}")
