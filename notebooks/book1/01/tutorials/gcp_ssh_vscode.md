# Using GCP via SSH + VScode

First [setup your GCP account](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/ssh_tunnels_and_how_to_dig_them.ipynb#scrollTo=TLtWT8vn-Eyh) and upload your ssh keys.
Next you should connect to your GCP VM via VScode:
just click on the green icon on the lower left corner, open a new ssh host, and type `username@ip-address`. Once connected, you can clone your github repo, edit your source code in a proper IDE, and [open a jupyter notebook inside VScode](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) for interactive development. (Use `%run foo.py` to run your script file; when it's done, all the global variables it created will be accessible from the notebook.) When you're done developing, save all your code back to github. (Artefacts can be stored using Google Cloud storage buckets.)
See the screenshot below for an example.
You can also [open tensorboard inside VScode](https://devblogs.microsoft.com/python/python-in-visual-studio-code-february-2021-release/#python-extension-updates), and it will handle port forwarding, etc. 

![](https://github.com/probml/probml-notebooks/raw/main/images/vscode-ssh.png)

