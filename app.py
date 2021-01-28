import os
from flask import Flask, request
from util.process_video_utils import save_uploaded_video, process_video, load_json_results, cleanup_files

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER') or os.path.join('static', 'videos')
RESULTS_FILE_PATH = os.getenv('RESULT_FILE_PATH') or os.path.join('static', 'results')
CONFIG_FILE = os.getenv('CONFIG_FILE') or os.path.join('helpers', 'config.json')

# preprocessing
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULTS_FILE_PATH):
    os.makedirs(RESULTS_FILE_PATH)

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process():
    video_file = save_uploaded_video(request.files, UPLOAD_FOLDER)
    results_file = process_video(video_file, RESULTS_FILE_PATH, CONFIG_FILE)
    results = load_json_results(results_file)
    cleanup_files(video_file, results_file)
    return results
