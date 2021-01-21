import json
import os

from flask import current_app

from detectors import VideoDetector

CONFIG_FILENAME = 'config.json'
RESULT_FILENAME = 'result.json'

CONFIG_FILE_PATH = os.path.join('helpers', 'yolo', CONFIG_FILENAME)
RESULT_FILE_PATH = os.path.join('static', 'results', RESULT_FILENAME)


def process_video(video_path, user_config):
    print('Procesando video utilizando YOLOv3...')

    print('Cargando configuraciones...')
    config = load_config(user_config)
    d = VideoDetector(config_dict=config)

    print('Cargando video...')
    d.load_file(video_path)

    print('Procesando video...')
    d.process()

    print('Guardando los resultados...')
    result_file = os.path.join(current_app.config['BASE_DIR'], CONFIG_FILE_PATH)
    d.write_json_to_file(result_file)

    # subir a base de datos

    print('Listo!!!')


def load_config(user_config):
    config_file = os.path.join(current_app.config['BASE_DIR'], CONFIG_FILE_PATH)

    # load base config
    with open(config_file, 'r') as f:
        config = json.load(f)

    # load user config
    for key in user_config:
        config[key] = user_config[key]

    return config
