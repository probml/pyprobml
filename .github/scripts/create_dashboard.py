import os
from glob import glob

statuses = glob("workflow_testing_indicator/notebooks/*/*/*.png")
base_url = "https://github.com/patel-zeel/pyprobml/tree/"

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
    for status in statuses:
        job = status.split("/", 2)[-1].split(".")[0]
        url = os.path.join(base_url, status)
        if os.path.exists(status.replace(".png", ".log")):
            log = os.path.join(base_url, status.replace(".png", ".log"))
            log_counter += 1
        else:
            log = "-"
        f.write(f"| [{job}]({url}) | ![{job}]({url}) | [log]({log}) |\n")
        file_counter += 1
    f.write(f"\n")
    f.write(f"## Summary\n")
    f.write(f"\n")
    f.write(f"In total, {file_counter} jobs were tested.\n")
    f.write(f"{log_counter} jobs failed.\n")
