# Parse log files from caliban_save_logs
# Authors: Guy Gur-Ari, Kevin Murphy


from absl import app
from absl import flags
 
import json
import pandas
import glob
 
FLAGS = flags.FLAGS
 
flags.DEFINE_string("files", "*.json", "Log files to load")
 
 
def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
 
  all_entries = []
  
  def flatten_json_payload(entry):
    if not "jsonPayload" in entry:
      return entry
 
    flat_entry = dict(entry)
    flat_entry.update(entry["jsonPayload"])
    del flat_entry["jsonPayload"]
    return flat_entry
 
  for filename in glob.glob(FLAGS.files):
    with open(filename, 'r') as f:
      entries = json.load(f)
      assert isinstance(entries, list)
      flat_entries = [flatten_json_payload(entry) for entry in entries]
      all_entries.extend(flat_entries)
 
  df = pandas.DataFrame(all_entries)
  print(df)
 
 
if __name__ == '__main__':
  app.run(main)