from glob import glob
import nbformat

# this packages need to passed by user
INSTALLED_MODULES = {
    "hashlib",
    "argparse",
    "dataclasses",
    "pytest_forked",
    "bleach",
    "_testimportmultiple",
    "imaplib",
    "IPython",
    "_pickle",
    "widgetsnbextension",
    "__future__",
    "uuid",
    "lzma",
    "webbrowser",
    "decimal",
    "backcall",
    "sysconfig",
    "nntplib",
    "sre_compile",
    "site",
    "asyncore",
    "blackd",
    "requests",
    "string",
    "fcntl",
    "weakref",
    "copyreg",
    "resource",
    "PIL",
    "xdrlib",
    "wheel",
    "typing_extensions",
    "_strptime",
    "platform",
    "six",
    "threadpoolctl",
    "_testmultiphase",
    "codecs",
    "ensurepip",
    "_ssl",
    "html",
    "_json",
    "sndhdr",
    "_multibytecodec",
    "nbclient",
    "kiwisolver",
    "graphviz",
    "charset_normalizer",
    "imp",
    "_compat_pickle",
    "doctest",
    "colorsys",
    "curses",
    "_multiprocessing",
    "psutil",
    "fileinput",
    "termios",
    "argon2",
    "contextvars",
    "_bz2",
    "logging",
    "xmlrpc",
    "mypy_extensions",
    "pandocfilters",
    "distutils",
    "numbers",
    "sre_parse",
    "webencodings",
    "_bootlocale",
    "macpath",
    "black",
    "smtpd",
    "traitlets",
    "zipapp",
    "ipython_genutils",
    "gzip",
    "keyword",
    "ipywidgets",
    "symbol",
    "tomli",
    "fastjsonschema",
    "mmap",
    "_dummy_thread",
    "pyparsing",
    "stringprep",
    "modulefinder",
    "binascii",
    "_osx_support",
    "gettext",
    "pydoc",
    "re",
    "pipes",
    "dis",
    "operator",
    "_markupbase",
    "execnet",
    "ftplib",
    "wcwidth",
    "_collections_abc",
    "netrc",
    "crypt",
    "_sysconfigdata_x86_64_conda_cos7_linux_gnu",
    "entrypoints",
    "nbconvert",
    "asynchat",
    "test",
    "warnings",
    "_codecs_hk",
    "send2trash",
    "enum",
    "threading",
    "plistlib",
    "concurrent",
    "_sysconfigdata_powerpc64le_conda_cos7_linux_gnu",
    "pydoc_data",
    "xxlimited",
    "_pytest",
    "profile",
    "blib2to3",
    "typed_ast",
    "datetime",
    "ipaddress",
    "posixpath",
    "_testbuffer",
    "opcode",
    "pytest_timeout",
    "_sysconfigdata_s390x_conda_cos7_linux_gnu",
    "pathspec",
    "cmd",
    "tracemalloc",
    "jupyterlab_pygments",
    "parso",
    "numpy",
    "matplotlib",
    "locale",
    "pvectorc",
    "aifc",
    "jax",
    "pylab",
    "pytest",
    "urllib",
    "_pyio",
    "os",
    "telnetlib",
    "tty",
    "compileall",
    "pyrsistent",
    "_sysconfigdata_i686_conda_cos6_linux_gnu",
    "_sysconfigdata_x86_64_conda_cos6_linux_gnu",
    "_sha512",
    "packaging",
    "socket",
    "wave",
    "jsonschema",
    "pickle",
    "binhex",
    "opt_einsum",
    "_datetime",
    "_sysconfigdata_m_linux_x86_64-linux-gnu",
    "importlib_metadata",
    "click",
    "readline",
    "statistics",
    "_weakrefset",
    "pstats",
    "antigravity",
    "_ctypes",
    "flatbuffers",
    "select",
    "uu",
    "syslog",
    "cffi",
    "pkgutil",
    "xml",
    "soupsieve",
    "pygments",
    "_lsprof",
    "calendar",
    "mistune",
    "fontTools",
    "ipykernel_launcher",
    "signal",
    "dummy_threading",
    "attr",
    "struct",
    "wsgiref",
    "selectors",
    "_sysconfigdata_aarch64_conda_linux_gnu",
    "pkg_resources",
    "_ctypes_test",
    "venv",
    "cgi",
    "cgitb",
    "defusedxml",
    "csv",
    "email",
    "tokenize",
    "abc",
    "2dd510b5c3364608e57a__mypyc",
    "qtconsole",
    "testbook",
    "pickletools",
    "_sitebuiltins",
    "pprint",
    "_sqlite3",
    "inspect",
    "socketserver",
    "jedi",
    "tornado",
    "pycparser",
    "timeit",
    "pathlib",
    "ssl",
    "mailcap",
    "http",
    "random",
    "cProfile",
    "_sha1",
    "pickleshare",
    "difflib",
    "idna",
    "_sysconfigdata_x86_64_apple_darwin13_4_0",
    "optparse",
    "token",
    "tempfile",
    "_sysconfigdata_aarch64_conda_cos7_linux_gnu",
    "shelve",
    "functools",
    "iniconfig",
    "_py_abc",
    "runpy",
    "jaxlib",
    "contextlib",
    "ptyprocess",
    "traceback",
    "configparser",
    "tabnanny",
    "debugpy",
    "bisect",
    "urllib3",
    "py_compile",
    "_codecs_tw",
    "prometheus_client",
    "collections",
    "pip",
    "typing",
    "symtable",
    "_threading_local",
    "fractions",
    "glob",
    "pyclbr",
    "platformdirs",
    "secrets",
    "zipfile",
    "_sha3",
    "textwrap",
    "reprlib",
    "certifi",
    "decorator",
    "scipy",
    "py",
    "importlib_resources",
    "array",
    "ctypes",
    "matplotlib_inline",
    "ntpath",
    "poplib",
    "trace",
    "_crypt",
    "tarfile",
    "types",
    "xdist",
    "_xxtestfuzz",
    "dateutil",
    "jinja2",
    "genericpath",
    "_hashlib",
    "grp",
    "formatter",
    "spwd",
    "notebook",
    "nis",
    "_codecs_cn",
    "cycler",
    "jupyter_client",
    "cmath",
    "_testcapi",
    "getpass",
    "_csv",
    "base64",
    "mimetypes",
    "_black_version",
    "lib2to3",
    "qtpy",
    "sunau",
    "rlcompleter",
    "_struct",
    "setuptools",
    "smtplib",
    "queue",
    "sqlite3",
    "_sysconfigdata_s390x_conda_linux_gnu",
    "filecmp",
    "_socket",
    "heapq",
    "_cffi_backend",
    "pexpect",
    "tinycss2",
    "parser",
    "_opcode",
    "tokenize_rt",
    "_codecs_iso2022",
    "jupyter_core",
    "dbm",
    "fnmatch",
    "pyexpat",
    "bs4",
    "_distutils_hack",
    "_lzma",
    "pytz",
    "getopt",
    "_elementtree",
    "_sysconfigdata_x86_64_conda_linux_gnu",
    "_queue",
    "_codecs_kr",
    "sched",
    "_blake2",
    "jupyter_console",
    "unicodedata",
    "quopri",
    "_asyncio",
    "shlex",
    "h5py",
    "json",
    "tkinter",
    "stat",
    "_curses_panel",
    "nturl2path",
    "asyncio",
    "subprocess",
    "get_installed_packegs",
    "unittest",
    "bdb",
    "turtledemo",
    "_codecs_jp",
    "chunk",
    "sre_constants",
    "importlib",
    "_contextvars",
    "imghdr",
    "zipp",
    "_curses",
    "_bisect",
    "attrs",
    "_heapq",
    "ossaudiodev",
    "ipykernel",
    "pluggy",
    "encodings",
    "pdb",
    "audioop",
    "seaborn",
    "_pyrsistent_version",
    "codeop",
    "jupyterlab_widgets",
    "mailbox",
    "math",
    "_tkinter",
    "bz2",
    "prompt_toolkit",
    "terminado",
    "_compression",
    "jupyter",
    "pty",
    "idlelib",
    "joblib",
    "hmac",
    "_sha256",
    "markupsafe",
    "copy",
    "turtle",
    "_md5",
    "io",
    "cached_property",
    "this",
    "multiprocessing",
    "pandas",
    "zmq",
    "ast",
    "zlib",
    "code",
    "shutil",
    "absl",
    "_argon2_cffi_bindings",
    "nbformat",
    "_random",
    "_pydecimal",
    "nest_asyncio",
    "sklearn",
    "_posixsubprocess",
    "linecache",
    "_decimal",
}


def get_installed_modules(installed_packages=INSTALLED_MODULES):
    # Special cases
    special_modules = set(["mpl_toolkits", "itertools", "time", "sys", "d2l", "augmax"])
    return special_modules.union(installed_packages)


def get_try_except_module(line):
    line = line.rstrip()
    import_kw = None

    if line.startswith(" ") and line.lstrip().startswith("import"):
        import_kw = "import "
    elif line.startswith(" ") and line.lstrip().startswith("from") and "import" in line:
        import_kw = "from "

    if import_kw:
        module = line.lstrip()[len(import_kw) :].split(" ")[0].split(".")[0]
        return module


def get_simply_imported_module(line):
    line = line.rstrip()
    import_kw = None

    if line.startswith("import "):
        import_kw = "import "
    elif line.startswith("from ") and "import" in line:
        import_kw = "from "

    if import_kw:
        module = line[len(import_kw) :].split(" ")[0].split(".")[0]
        return module


def wrap_line_with_try_accept(line, module):

    transformed_modules = {
        "PIL": "pillow",
        "tensorflow_probability": "tensorflow-probability",
        "sklearn": "scikit-learn",
        "pl_bolts": "lightning-bolts",
        "skimage": "scikit-image",
        "cv2": "opencv-python",
        "tensorflow_datasets": "tensorflow tensorflow_datasets",
        "umap":"umap-learn"
    }
    f"""
    check if import {module} is in given line: {line}
    if present, then return {line} wrapped with try...except
    """
    line = line.rstrip()
    if module in transformed_modules:
        module = transformed_modules[module]
    try_except_line = f"try:\n    {line}\nexcept ModuleNotFoundError:\n    %pip install -qq {module}\n    {line}"
    return try_except_line


def wrap_try_accept_in_code(code):
    lines = code.split("\n")
    try_except_modules = set(map(get_try_except_module, lines))
    present_modules = try_except_modules.union(get_installed_modules(installed_packages=INSTALLED_MODULES))

    for line_no, line in enumerate(lines):
        module = get_simply_imported_module(line)
        if module and module not in present_modules:
            updated_line = wrap_line_with_try_accept(line, module)
            print(updated_line)
            lines[line_no] = updated_line
            present_modules.add(module)
    code = "\n".join(lines)
    return code


def remove_superimport(code):
    lines = code.split("\n")
    updated_code = "\n".join(list(map(lambda line: line.replace("import superimport", ""), lines)))
    return updated_code

def remove_pyprobml(code):
    code = code.replace("from pyprobml_utils import save_fig", "from probml_utils import savefig")
    code = code.replace("%pip install pyprobml_utils", "%pip install git+https://github.com/probml/probml-utils.git")
    code = code.replace("import pyprobml_utils as pml", "import probml_utils as pml")
    return code
    
def apply_fun_to_notebook(notebook, fun):
    """
    fun should take one argument: code
    """
    nb = nbformat.read(notebook, as_version=4)
    for cell in nb.cells:
        code = cell["source"]
        updated_code = fun(code)
        if updated_code != code:
            cell["source"] = updated_code
            nbformat.write(nb, notebook)


if __name__ == "__main__":
    # Load notebooks
    notebooks1 = glob("notebooks/book1/*/*.ipynb")
    notebooks2 = glob("notebooks/book2/*/*.ipynb")
    notebooks = notebooks1 + notebooks2
    
    #get IGNORE_LIST of notebooks
    IGNORE_LIST = []
    with open("internal/ignored_notebooks.txt") as fp:
        ignored_notebooks = fp.readlines()
        for nb in ignored_notebooks:
            IGNORE_LIST.append(nb.strip().split("/")[-1])

    def in_ignore_list(nb_path):
        nb_name = nb_path.split("/")[-1]
        return nb_name in IGNORE_LIST

    print(f"{len(IGNORE_LIST)} notebooks ignored")
    notebooks = list(filter(lambda nb: not in_ignore_list(nb), notebooks))

    for notebook in notebooks:
        print(f"******* {notebook} *******")
        apply_fun_to_notebook(notebook, remove_superimport)
        apply_fun_to_notebook(notebook, wrap_try_accept_in_code)
