import pandas as pd
import toml

## Config
with open("internal/contributors/contributors.toml") as f:
    config = toml.load(f)

## Get all contributors from a repo
api_call = f"https://api.github.com/repos/{config['owner']}/{config['repo']}/contributors?per_page=100"
contributors = pd.read_json(api_call).set_index("login")

## Fetch all PRs from a repo
page_no = 1
pr_df_list = []
while True:
    api_call = (
        f"https://api.github.com/repos/{config['owner']}/{config['repo']}/pulls?state=all&per_page=100&page={page_no}"
    )
    df = pd.read_json(api_call)
    if len(df) == 0:
        break
    pr_df_list.append(df)
    page_no += 1
pull_requests = pd.concat(pr_df_list)

## Get count of PRs per contributor

pull_requests["login"] = pull_requests["user"].apply(lambda x: x["login"])
contributor_pr = pull_requests.groupby("login").agg({"url": len}).sort_values(by="url", ascending=False)
contributor_pr.rename(columns={"url": "Number of PRs"}, inplace=True)

# Filtering
atleast_2_pr = contributor_pr[contributor_pr["Number of PRs"] >= config["min_PRs"]]
atleast_2_commits = contributors[contributors["contributions"] >= config["min_commits"]]

union_users = atleast_2_pr.index.union(atleast_2_commits.index).to_list()
union_users = sorted(union_users, key=lambda x: x.lower())

## Create a dashboard
def get_href_user(user):
    return f"[{user}](https://github.com/{user})"


dashboard = pd.DataFrame(index=union_users)
dashboard["login"] = dashboard.index
dashboard["Avatar"] = dashboard.login.apply(
    lambda x: f'<img width="{config["fig_width"]}" alt="image" src="https://github.com/{x}.png">'
)
dashboard["Contributor"] = dashboard.login.apply(get_href_user)
md_strings = dashboard[["Avatar", "Contributor"]].T.to_markdown().split("\n")

# Little formatting
print("| " + md_strings[2].split("|", 2)[-1])
print("| " + md_strings[1].split("|", 2)[-1].replace("-|", ":|"))
print("| " + md_strings[3].split("|", 2)[-1])
