import os
from glob import glob
parser = argparse.ArgumentParser(description="delete firestore")
parser.add_argument("-user_name", "--user_name", type=str, help="")
args = parser.parse_args()

statuses = glob("workflow_testing_indicator/notebooks/*/*/*.png")
user = args.user_name

base_url = f"https://github.com/{user}/pyprobml/tree/"
get_url = lambda x: f'<img width="20" alt="image" src=https://raw.githubusercontent.com/{user}/pyprobml/{x}>'
get_nb_url = lambda x: os.path.join(base_url, "master", x.split("/", 1)[-1].replace(".png", ".ipynb"))

# sort statuses
def sort_key(x):
    parts = x.split("/")
    return (parts[-3], parts[-2])


statuses = sorted(statuses, key=sort_key)

# write an md file
log_counter = 0
file_counter = 0
with open("workflow_testing_indicator/README.md", "w") as f:
    f.write(f"# PyProbML status\n")
    f.write(f"\n")
    f.write(f"## Status\n")
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
    final_log = f"In total, {file_counter} jobs were tested.\n{log_counter} jobs failed.\n"
    f.write(final_log)
    print(final_log)
