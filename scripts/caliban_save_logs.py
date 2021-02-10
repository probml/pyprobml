
# https://github.com/google/caliban
# Authors: Guy Gur-Ari, Kevin Murphy

'''
Download log files (in JSON format) from GCP after running caliban jobs.
Creates files foo_x.log and foo_x.config for each job number x
matching specified xgroup. 

The command 'caliban status' generates something like this

most recent 8 jobs for user kpmurphy:
I0209 11:36:10.360710 4319092160 cli.py:77] xgroup my_experiment2:
I0209 11:36:10.361803 4319092160 cli.py:85] docker config 19: job_mode: CPU, build url: ~/caliban, extra dirs: None
I0209 11:36:10.361921 4319092160 cli.py:94]   experiment id 30: mnist2.py  --learning_rate 0.01 --width 10
I0209 11:36:10.363482 4319092160 cli.py:98]     job 36       SUCCEEDED     CAIP 2021-02-08 21:00:53 container: gcr.io/probml/339ce8125679:latest name: caliban_kpmurphy_20210208_210052_1
...

I0209 11:36:10.366102 4319092160 cli.py:77] xgroup jax:
I0209 11:36:10.366506 4319092160 cli.py:85] docker config 20: job_mode: GPU, build url: ~/caliban/Jax, extra dirs: None
I0209 11:36:10.366576 4319092160 cli.py:94]   experiment id 36: jax2.py  --ndims 100
I0209 11:36:10.366969 4319092160 cli.py:98]     job 42       SUCCEEDED     CAIP 2021-02-08 22:30:54 container: gcr.io/probml/d7c6d9760538:latest name: caliban_kpmurphy_20210208_223053_1
I0209 11:36:12.744846 4319092160 cli.py:98]     job 43       SUCCEEDED     CAIP 2021-02-08 22:42:04 container: gcr.io/probml/11ae4953adf3:latest name: caliban_kpmurphy_20210208_224202_1

We get the status for a specific xgroup, and then parse the output to get the job names.
Once we have the job names, we can extract logs using a command like

gcloud logging read 'resource.labels.job_id="caliban_kpmurphy_20210208_194505_1"'


To get the configuration (flags) used for each job, we use can use a command like this

gcloud ai-platform jobs describe caliban_kpmurphy_20210208_194505_1 --format=json


'''


import re
import subprocess
 
from absl import app
from absl import flags
 
 
FLAGS = flags.FLAGS
 
flags.DEFINE_string('xgroup', None, 'The experiment group to query')
flags.DEFINE_integer('limit', 10000, 'Max number of log entries per job')
 
 
def main(_):
  assert FLAGS.xgroup
  proc = subprocess.Popen(
    ['caliban', 'status', '--xgroup', FLAGS.xgroup],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE)
    
  for line in iter(proc.stderr):
    line = str(line, "utf-8").strip()
    m = re.search(r'name: (.*)', line)
    if m:
      name = m.group(1)
      print(name)
 
      rc = subprocess.call(
          "gcloud logging read 'resource.labels.job_id=\"%s\"' --format json --limit %d > %s.log" % (name, FLAGS.limit, name),
          shell=True)
      print("rc:", rc)
      
      rc = subprocess.call(
          "gcloud  ai-platform jobs describe %s --format json > %s.config" % (name, name),
          shell=True)
      print("rc:", rc)
 
 
if __name__ == '__main__':
  app.run(main)
