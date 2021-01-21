"""
Configuration class to be shared between detector classes in the module.
Author: Carlos Cuevas
@cavesdev
February 2020
"""
import json


class Config:
    """
    Configuration class that shares config variables with the detector classes.
    """

    # default configurations
    __conf = {
        'fps': 30,
        'frame-skip': 30,
        'model-cfg': 'yolov3.cfg',
        'model-weights': 'yolov3.weights',
        'model-classes': 'coco.names',
        'output-filename': 'yolo_out_py',
        'confidence-threshold': 0.5,
        'nms-threshold': 0.4,
        'input-width': 416,
        'input-height': 416,
        'show': False,
        'save': False,
        'gpu': False
    }

    # configurations that can be changed by the user
    __setters = [
        'fps',
        'frame-skip',
        'model-cfg',
        'model-weights',
        'model-classes',
        'output-filename',
        'show',
        'save',
        'gpu'
    ]

    @staticmethod
    def get(name):
        """
        Get configuration variable
        :param name: the configuration value to get
        """
        return Config.__conf[name]

    @staticmethod
    def set(name, value):
        """
        Set the value of the given configuration variable if it can be changed by the user (if it is in setters)
        :param name: the name of the configuration variable to set/change
        :param value: the value to put on the configuration variable
        """
        if name in Config.__setters:
            Config.__conf[name] = value
        else:
            raise NameError('Name not accepted in set() method.')

    @staticmethod
    def load_from_file(filename):
        """
        Load configurations from a JSON file.
        :param filename: name of the JSON file to load.
        """
        with open(filename, "r") as f:
            data = json.load(f)

        Config.load_from_dict(data)

    @staticmethod
    def load_from_dict(data):
        """
        Load configurations from a dictionary
        :param data: dictionary object with configurations variables to set
        """
        for key in data:
            Config.set(key, data[key])
