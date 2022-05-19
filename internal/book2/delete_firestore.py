# command usage:
"""
Delete all keys: python internal/book2/delete_firestore.py -key "../key_probml_gcp.json" -level1 figures -level2 book2 -level3 figures
Delete 1 key: python internal/book2/delete_firestore.py -key "../key_probml_gcp.json" -level1 figures -level2 book2 -level3 figures -key_del 10.1
"""
from probml_utils.url_utils import create_firestore_db
import argparse

parser = argparse.ArgumentParser(description="delete firestore")
parser.add_argument("-key", "--key", type=str, help="")
parser.add_argument("-level1", "--level1", type=str, help="")
parser.add_argument("-level2", "--level2", type=str, help="")
parser.add_argument("-level3", "--level3", type=str, help="")
parser.add_argument("-key_del", "--key_del", type=str, help="")
args = parser.parse_args()

# arguments
key_path = args.key
level1 = args.level1
level2 = args.level2
level3 = args.level3
key_to_delete = args.key_del


db = create_firestore_db(key_path)
ref = db.collection(level1).document(level2).collection(level3)
n_doc = len(ref.get())
print(f"Deleteing {n_doc} documents from {level1}/{level2}/{level3}")

if not key_to_delete:
    for doc in ref.get():
        ref.document(doc.id).delete()
    print(f"{n_doc} documents deleted")

else:
    ref.document(key_to_delete).delete()
    print(f"{key_to_delete} deleted")
