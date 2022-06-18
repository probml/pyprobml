import os
from glob import glob
import argparse

parser = argparse.ArgumentParser(description="create dashboard")
parser.add_argument("-user_name", "--user_name", type=str, help="")
args = parser.parse_args()
user, repo = args.user_name.split("/") # github.repository gives owner/repo

print(f"\n**** creating dashboard on {user}/{repo} *********\n")
base_url = f"https://github.com/{user}/{repo}/tree/"
get_url = lambda x: f'<img width="20" alt="image" src=https://raw.githubusercontent.com/{user}/{repo}/{x}>'
get_nb_url = lambda x: os.path.join(base_url, "master", x.split("/", 1)[-1].replace(".png", ".ipynb"))

# sort statuses
def sort_key(x):
    parts = x.split("/")
    return (parts[-3], parts[-2])

# write an md file
with open("workflow_testing_indicator/README.md", "w") as f:
    for book_no in [1,2]:
        log_counter = 0
        file_counter = 0

        statuses = glob(f"workflow_testing_indicator/notebooks/book{book_no}/*/*.png")
        print(f"**** {len(statuses)} statuses found in Book{book_no}****")
        statuses = sorted(statuses, key=sort_key)
        f.write(f"\n# Book {book_no}: PyProbML status\n")
        f.write(f"\n")
        f.write(f"\n")
        f.write(f"| Job | Status | Log |\n")
        f.write(f"| --- | --- | --- |\n")
        passing = []
        failing = []
        for status in statuses:
            job = status.split("/", 2)[-1].split(".")[0]
            url = get_url(status)
            url_to_nb = get_nb_url(status)
            if os.path.exists(status.replace(".png", ".log")):
                log = os.path.join(base_url, status.replace(".png", ".log"))
                failing.append(f"| [{job}]({url_to_nb}) | {url} | [log]({log}) |\n")
                log_counter += 1
            else:
                log = "-"
                passing.append(f"| [{job}]({url_to_nb}) | {url} | [log]({log}) |\n")
            file_counter += 1
        for entry in passing+failing:
            f.write(entry)
        f.write(f"\n")
        f.write(f"## Summary\n")
        f.write(f"\n")
        final_log = f"Book{book_no}: In total, {file_counter} jobs were tested.\n{log_counter} jobs failed.\n"
        f.write(final_log)
        print(final_log)