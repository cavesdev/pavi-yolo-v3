import os
from flask import Flask, request
from util.process_video_utils import save_uploaded_video, process_video

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER') or os.path.join('static', 'videos')
RESULT_FILE_PATH = os.getenv('RESULT_FILE_PATH') or os.path.join('static', 'results')
CONFIG_FILE = os.getenv('CONFIG_FILE') or os.path.join('helpers', 'config.json')

# preprocessing
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FILE_PATH):
    os.makedirs(RESULT_FILE_PATH)

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process():
    video_path = save_uploaded_video(request.files, UPLOAD_FOLDER)
    process_video(video_path, RESULT_FILE_PATH, CONFIG_FILE)
    return {
        "message": "video processed"
    }
