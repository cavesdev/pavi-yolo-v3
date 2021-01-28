import json
import os
import uuid

from flask import abort
from detectors import VideoDetector

SUPPORTED_VIDEO_FORMATS = ['mp4']


def save_uploaded_video(files, upload_folder):
    if 'video' not in files:
        abort(400, description="Video data not found in request.")

    video = files['video']

    if video.filename == '':
        abort(400, description="Video file not sent.")

    if video and supported_file(video.filename):
        _, ext = os.path.splitext(video.filename)
        filename = str(uuid.uuid1()) + ext
        filepath = os.path.join(upload_folder, filename)
        video.save(filepath)
        return filepath
    else:
        abort(415, description="Video format not supported.")


def supported_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in SUPPORTED_VIDEO_FORMATS


def process_video(video_path, result_file_path, config_file):
    print('Procesando video utilizando YOLOv3...')

    print('Cargando configuraciones...')
    if os.path.exists(config_file):
        config = load_config(config_file)
    else:
        config = None

    d = VideoDetector(config_dict=config)

    print('Cargando video...')
    d.load_file(video_path)

    print('Procesando video...')
    d.process()

    print('Guardando los resultados...')
    filename, _ = os.path.splitext(os.path.basename(video_path))
    results_file = os.path.join(result_file_path, filename + '.json')
    d.write_json_to_file(results_file)

    print('Listo!!!')
    return results_file


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def load_json_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


def cleanup_files(video_file, results_file):
    if os.path.exists(video_file):
        os.remove(video_file)
    if os.path.exists(results_file):
        os.remove(results_file)
    return
