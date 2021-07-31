import sys
import subprocess
import pkg_resources
import requests
import pipreqs
import inspect
import re
import logging
import os

# based on
# https://stackoverflow.com/questions/44210656/how-to-check-if-a-module-is-installed-in-python-and-if-not-install-it-within-t
# https://stackoverflow.com/questions/52311738/get-name-from-python-file-which-called-the-import
# https://gist.github.com/gene1wood/9472a9d0dffce1a56d6e796afc6539b8
# https://stackoverflow.com/questions/8718885/import-module-from-string-variable


def get_packages_from_txt(file, dim="="):

    packages_string = open(file).read()
    if dim:
        packages = {
            c.split(dim)[0]: c.split(dim)[1] for c in packages_string.split("\n") if c
        }
    else:
        packages = {c: True for c in packages_string.split("\n") if c}
    return packages


def install_if_missing(packages_names, verbose=False):

    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = packages_names - installed
    if missing:
        python3 = sys.executable
        if verbose:
            subprocess.check_call([python3, "-m", "pip", "install", *missing])
        else:
            subprocess.check_call(
                [python3, "-m", "pip", "install", *missing], stdout=subprocess.DEVNULL
            )


def get_match_list(the_string, the_regex_pattern, guard="#"):
    if not the_string or the_string == "":
        return None
    re_string = re.compile(the_regex_pattern)
    matches_itr = re.finditer(re_string, the_string)
    matches_list = list(matches_itr)
    matches_list = [m for m in matches_list if the_string[m.span()[0] - 1] != guard]
    return matches_list


def get_imports(
    file_string=None, patterns=[r"^import (.+)$", r"^from ((?!\.+).*?) import (?:.*)$"]
):
    matches = []
    for p in patterns:
        strings = file_string.split("\n")
        for s in strings:
            re_matches = get_match_list(s, p)
            if re_matches:
                for m in re_matches:
                    the_string = m.group()
                    if the_string.startswith("from"):
                        i = the_string.find("import")
                        name = the_string[5:i]

                    else:
                        name = the_string.replace("import ", "")
                    matches.append(name)
    return set(matches)


def check_if_package_on_pypi(packages_name):
    response = requests.get(f"https://pypi.python.org/pypi/{packages_name}/json")
    if response.status_code == 200:
        meta = response.json()
        name = meta["info"]["name"]
        return True, name, meta
    else:
        return False, None, None


def import_module(module_name, verbose=False):
    try:
        # because we want to import using a variable, do it this way
        module_obj = __import__(module_name)
        # create a global object containging our module
        globals()[module_name] = module_obj
    except ImportError as e:
        if verbose:
            sys.stderr.write(
                f"ERROR: superimport : missing python module: {module_name} \nTrying try to install automatcially\n"
            )
        raise e


mapper = pipreqs.__path__[0] + "/mapping"


mapping = get_packages_from_txt(mapper, ":")
stdlib_path = pipreqs.__path__[0] + "/stdlib"
stdlib = get_packages_from_txt(stdlib_path, "")
mapping2 = get_packages_from_txt("./superimport/mapping2", ":")

mapping = {**mapping, **mapping2}  # adding two dictionaries

gnippam = {v: k for k, v in mapping.items()}  # reversing the mapping

if __name__ != "__main__":
    for frame in inspect.stack()[1:]:
        if frame.filename[0] != "<":
            fc = open(frame.filename).read()
            fc = fc.replace("import superimport\n", "")
            matches = get_imports(fc)
            for package in matches:
                try:
                    import_module(package, True)
                except Exception as e:
                    if package in mapping:
                        install_if_missing({gnippam[package]}, True)
                    else:
                        logging.warning("Package was not found in the reverse index.")
                        status, name, meta = check_if_package_on_pypi(package)
                        if status:
                            logging.info(
                                f"Package{name} was found on PyPi\nNow installing {name}"
                            )
                            install_if_missing({package}, True)
                        else:
                            logging.warning(
                                f"Failed to install {package} automatically"
                            )
            break
