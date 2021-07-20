# Parse log files from caliban_save_logs
# Authors: Guy Gur-Ari, Kevin Murphy


from absl import app
from absl import flags
 
import json
import pandas as pd
import glob
 
FLAGS = flags.FLAGS
 
flags.DEFINE_string("logdir", "", "Directory containing log files")
 
def flatten_json_payload(entry):
  if not "jsonPayload" in entry:
    return entry
  flat_entry = dict(entry)
  flat_entry.update(entry["jsonPayload"])
  del flat_entry["jsonPayload"]
  return flat_entry

def json_file_to_pandas(filename):
  with open(filename, 'r') as f:
    entries = json.load(f)
    if isinstance(entries, list):
      flat_entries = [flatten_json_payload(entry) for entry in entries]
    else:
      flat_entries = [flatten_json_payload(entries)]
  df = pd.DataFrame(flat_entries)
  return df

def get_job_num_from_fname(fname):
  # Extract job number from filename
  # eg. '/content/gdrive/MyDrive/Logs/caliban_kpmurphy_20210208_194505_1.json' to 1
  parts = fname.split('.') # separate into filename and suffix
  body = parts[0]
  parts = body.split('_') # parse jobname into pieces
  job_num = parts[-1] # final piece is the number
  return int(job_num)

def json_dir_to_pandas(fnames):
  df_list = []
  for filename in fnames:
    print('reading ', filename)
    df = json_file_to_pandas(filename)
    num = get_job_num_from_fname(filename)
    df['job_num'] = num
    df = df.astype({'job_num': 'int32'})
    df_list.append(df)
  return pd.concat(df_list)

def parse_logs(logdir):
  fnames = glob.glob(f'{logdir}/*.log')
  return json_dir_to_pandas(fnames)

def parse_configs(logdir):
  fnames = glob.glob(f'{logdir}/*.config')
  return json_dir_to_pandas(fnames)

def get_log_messages(df, job_num=None):
  '''Return list of log messages for this job'''
  if job_num:
    df = df[df.job_num == job_num]
  # messages are stored most recent first. We restore to chronological order.
  df = df[['timestamp', 'message']].copy()
  df['time'] = pd.to_datetime(df.timestamp)
  df = df.sort_values(by='time', ascending=True)
  messages = df.loc[:, ['message']].dropna()
  return messages.values

def get_args(df, job_num):
  '''Return list of arguments (flags) passed to this job'''
  dic = configs.loc[df.job_num==job_num,'trainingInput'].values[0]
  args = dic['args']
  return args


 
def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    configs_df = parse_configs(FLAGS.logdir)
    print(configs_df)
    logs_df = parse_logs(FLAGS.logdir)
    print(logs_df) 
  

 
if __name__ == '__main__':
  app.run(main)