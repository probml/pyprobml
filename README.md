# pyprobml
Python 3 code for "Machine learning: a probabilistic perspective" (http://people.cs.ubc.ca/~murphyk/MLbook/) v2.0.

You should run the code from the repo root ('pyprobml' directory)
using module execution, e.g.,

python -m examples.LMSdemo

To run all the examples you can use

ls examples/*.py | sed 's/\//./' | sed s/\.[^\.]*$// | xargs -n 1 python -m

from the repo root. This lists all the Python files under examples, turns the slash into a
period, then removes the .py extension and finally feeds each line into python -m. 


