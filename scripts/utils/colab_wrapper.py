file = 'chow_liu_tree_demo' # edit me
#file = 'datasaurus_dozen' # edit me

# specify location of contents
file_setup = f'{file}_setup.sh'
file_script = f'{file}.py'
url = 'https://raw.githubusercontent.com/probml/pyprobml/master/scripts/'
url_utils = f'{url}/pyprobml_utils.py'
url_setup = f'{url}/{file_setup}'
url_script = f'{url}/{file_script}'

# make directories
!mkdir figures
!mkdir scripts
%cd /content/scripts

# download what you need
!wget -q $url_utils
!wget -q $url_setup
!wget -q $url_script

# install any needed packages
!bash $file_setup
import pyprobml_utils as pml
import matplotlib.pyplot as plt

%run $file_script
plt.show() # force display figures
