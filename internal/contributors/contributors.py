import pandas as pd

## Config
owner = "probml"
repo = "pyprobml"
min_PRs = 2
min_commits = 2
fig_width = 50

## Get all contributors from a repo
api_call = f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=100"
contributors = pd.read_json(api_call).set_index("login")

## Fetch all PRs from a repo
page_no = 1
pr_df_list = []
while True:
    api_call = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all&per_page=100&page={page_no}"
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

# Printing
all_users = contributors.index.union(contributor_pr.index).to_list()
tmp_df = pd.DataFrame(index=all_users)
tmp_df["Number of PRs"] = contributor_pr["Number of PRs"].fillna(0)
tmp_df["contributions"] = contributors["contributions"].fillna(0)
print(f"# Contributors of {repo} are {len(all_users)}")
print(tmp_df.to_string())
print("#####################################################################")

# Filtering
atleast_n_pr = contributor_pr[contributor_pr["Number of PRs"] >= min_PRs]
atleast_n_commits = contributors[contributors["contributions"] >= min_commits]

union_users = atleast_n_pr.index.union(atleast_n_commits.index).to_list()
union_users = sorted(union_users, key=lambda x: x.lower())

## Create a dashboard
def get_href_user(user):
    return f"[{user}](https://github.com/{user})"


dashboard = pd.DataFrame(index=union_users)
dashboard["login"] = dashboard.index
dashboard["Avatar"] = dashboard.login.apply(
    lambda x: f'<img width="{fig_width}" alt="image" src="https://github.com/{x}.png">'
)
dashboard["Contributor"] = dashboard.login.apply(get_href_user)
md_strings = dashboard[["Avatar", "Contributor"]].T.to_markdown().split("\n")

# Print other stats
print("#####################################################################")
print(
    f"# Contributors with at least {min_PRs} PRs or at least {min_commits} commits are: {len(union_users)} out of {len(all_users)}"
)
print("#####################################################################")
print("\n\nCopy paste below string in the README.md file:\n\n")

# Little formatting
print("| " + md_strings[2].split("|", 2)[-1])
print("| " + md_strings[1].split("|", 2)[-1].replace("-|", ":|"))
print("| " + md_strings[3].split("|", 2)[-1])
