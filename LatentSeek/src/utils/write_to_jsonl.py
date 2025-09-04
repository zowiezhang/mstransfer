import json

def write_to_jsonl(lock, file_name, data):
    with lock:
        with open(file_name, 'a') as f:
            json.dump(data, f)
            f.write('\n')