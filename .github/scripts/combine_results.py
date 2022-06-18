import os
import re
import argparse

parser = argparse.ArgumentParser(description="combine results")
parser.add_argument("-user_name", "--user_name", type=str, help="")
args = parser.parse_args()
user, repo = args.user_name.split("/")

# Collect status, logs and figures
FOLDERS = ["workflow_testing_indicator", "auto_generated_figures"]
for folder in FOLDERS:
    if os.path.exists(folder):
        os.system("rm -rf " + folder)
    os.makedirs(folder)

try:
    # Read number of jobs
    with open(".github/workflows/notebooks.yml") as f:
        runners = re.findall("runner_id: (.*)", f.read())
        n_jobs = len(eval(runners[0]))

    print(f"{n_jobs} jobs found")

    repo_url = f"https://github.com/{user}/{repo}.git"
    for job in range(n_jobs):
        for folder in FOLDERS:
            branch = f"{folder}_{job}"
            if os.path.exists(branch):
                os.system(f"rm -rf {branch}")
            os.system(f"git clone --depth 1 --branch {branch} {repo_url} {branch}")
            os.system(f"cp -r {branch}/* {folder}")
            print(f"Copied {branch} to {folder}")
            os.system(f"rm -rf {branch}")
except:
    print("No runners found")
