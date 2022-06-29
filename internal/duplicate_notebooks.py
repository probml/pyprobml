from glob import glob
import os
import shutil
import regex as re
from pathlib import Path

book1_notebooks = glob("notebooks/book1/*/*.ipynb")
book2_notebooks = glob("notebooks/book2/*/*.ipynb")
misc_notebooks = glob("notebooks/misc/*.ipynb") + glob("notebooks/misc/*/*.ipynb")
base_url = "https://colab.research.google.com/github/probml/pyprobml/blob/master/"

print(len(book1_notebooks), len(book2_notebooks), len(misc_notebooks))

get_notebook_name = lambda notebook: notebook.split("/")[-1]

book1_notebooks_names = set(list(map(get_notebook_name, book1_notebooks)))
book2_notebooks_names = set(list(map(get_notebook_name, book2_notebooks)))
notebook_names = book1_notebooks_names.union(book2_notebooks_names)

def remove_duplicate_nb_by_name():
    for misc_notebook in misc_notebooks:
        notebook_name = get_notebook_name(misc_notebook)
        if notebook_name in notebook_names:
            print(f"{misc_notebook} is a duplicate")
            shutil.move(misc_notebook, f"deprecated/")
        
def get_path_nb(nb):
    for notebook in book1_notebooks:
        if get_notebook_name(notebook) == nb:
            return notebook

def copy_referred_nb():
    # Readme.md
    readme_files = glob("notebooks/book1/*/README.md")
    refered_nb = []
    copied_nb = []
    for readme_file in readme_files:
        print(f"************* {readme_file} **************")
        with open(readme_file, "r") as f:
            updated_flg = 0
            content = f.read()
            if "## Supplementary material" in content:
                new_content_lines = content.split("## Supplementary material")[0].split("\n")
                new_content_lines.append("## Supplementary material")
                content = content.split("## Supplementary material")[1]
                for line in content.split("\n"):
                    last_field = line.split("|")[-1]
                    if "Notebook" in last_field or "[d2lbook]" in last_field:
                        link = last_field.replace("[Notebook]", "").replace("[d2lbook]", "").replace("(", "").replace(")", "") #get link to nb
                        nb_name = link.split("/")[-1]
                        refered_nb.append(nb_name)
                        nb_misc_file = f"notebooks/misc/{nb_name}"
                        nb_dest = f"{readme_file.replace('README.md','')}{nb_name}"
                        if  nb_misc_file in misc_notebooks:
                            shutil.copy(nb_misc_file, nb_dest) #copy from misc to current chapter
                            print(f"{nb_misc_file} -> {nb_dest}")
                            line = line.replace(last_field,f"[{nb_name}]({os.path.join(base_url,nb_dest)})") #update the link
                            copied_nb.append(nb_misc_file) #track which nb are copied
                            updated_flg = 1
                        else:
                            curr_chapter_nb = glob(f"{readme_file.replace('README.md','')}*.ipynb")
                            #print(curr_chapter_nb)
                            # check if notebook in current chapter
                            if nb_dest in curr_chapter_nb:
                                line = line.replace(last_field,f"[{nb_name}]({os.path.join(base_url,nb_dest)})") #update the link
                                updated_flg = 1
                                print(f"{nb_dest} exists in current chapter")

                            #check if notebook is in different chapter
                            else:
                                nb_dest = get_path_nb(nb_name)
                                #print(nb_dest, book1_notebooks)
                                if nb_dest in book1_notebooks:
                                    updated_link = os.path.join(base_url, nb_dest)
                                    line = line.replace(last_field,f"[{nb_name}]({updated_link})")
                                    updated_flg = 1
                                    print(f"{nb_dest} exists in different chapter") 
                                    
                                else:
                                    print(f"{link} not in misc and not in current chapter!!")

                    new_content_lines.append(line)

        if updated_flg:
            with open(readme_file, "w") as f:
                f.write("\n".join(new_content_lines))
        #break

    return copied_nb
    


def delete_nb(notebook_list):
    [os.remove(nb) for nb in notebook_list]
    print(f"{len(notebook_list)} deleted!")

def store_copied_nb(notebooks,fname = "internal/ignored_notebooks.txt"):
    with open(fname,"w") as fp:
        [fp.write(nb+"\n") for nb in notebooks]

if __name__ == "__main__":
    print("main")
    # copied_nb = copy_referred_nb()
    # print(len(copied_nb), len(set(copied_nb)))
    # print(copied_nb[:4])
    # store_copied_nb(copied_nb)
    # delete_nb(set(copied_nb))

'''
# some issues
1. Needs to update probml-notebooks/ link to pyprobml/
'''
