import argparse
import json
import os

from pymongo import MongoClient

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='JSON a cargar en la base de datos')
args = vars(ap.parse_args())

json_file = args['input'].strip()

print('Subiendo los resultados a la base de datos...')
# load to mongodb
with open(json_file, "r") as f:
    data = json.load(f)

mongo_url = os.environ.get('MONGO_URI')
client = MongoClient(mongo_url)
db = client.pavi
videos = db.videos

video = videos.find_one({'filename': data['filename']})


def has_algorithm(video, data):
    algorithm = data['processing'][0]['algorithm']
    found = False
    found_index = 0
    index = 0
    found_index = 0
    for item in video['processing']:
        if item['algorithm'] == algorithm:
            found = True
            found_index = index
        index += 1
    return found, found_index


if video is not None:

    found, index = has_algorithm(video, data)

    if found:
        # replace algorithm data
        video['processing'][index] = data['processing'][0]
    else:
        # add algorithm data
        video['processing'] += data['processing']

    videos.replace_one(
        {'filename': data['filename']},
        video
    )
else:
    videos.insert_one(data)