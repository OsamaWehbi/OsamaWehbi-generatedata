import json

from src.apis.fed_sqlite import FedDB

db = FedDB("./perf.db")

data = db.get('mycifar', 'local_acc')[-1]
data = eval(data)
print(sorted(data.values())[:29])

















