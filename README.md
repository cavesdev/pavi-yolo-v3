# YOLOv3 processing API

API based around yolo-detection python package to process videos.

For the API, the usage is as follows:
- Specify any environment variables necessary in .env file on the root directory. Defaults are:
  - `UPLOAD_FOLDER` = ./static/videos
  - `RESULT_FILE_PATH` = ./static/results  
  - `CONFIG_FILE` = ./helpers/config.json
- Download the helper files mentioned below and specify their path in the configuration file.
- Install the dependencies `pip install -r requirements.txt`
- Run the Flask application `flask run` or `python3 app.py`

# yolo-detection
Python script using YOLOv3 algorithm and OpenCV to detect objects in images and video. 

To get the recommended helper files use the following commands:

```shell script
curl -O https://pjreddie.com/media/files/yolov3.weights
curl -O https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg\?raw=true
curl -O https://github.com/pjreddie/darknet/blob/master/data/coco.names\?raw=true
```

You can also find different YOLO versions available at [the author's website.](https://pjreddie.com/darknet/yolo/)

To import the detectors:
```python
from detectors import VideoDetector
from detectors import ImageDetector
```

The general process is:
```python
from detectors import VideoDetector
d = VideoDetector(config_file='config.json')
# or d = VideoDetector(config_dict={})
"""
Optional kwargs:
- config_file: a JSON file with your configurations
- config_dict: a dictionary with your configurations
Just one is accepted, in the order mentioned.
"""
d.load_file('filename.mp4')
d.process()
d.write_json_to_file('filename.json') # optional: write the detection results to a JSON file.
```

Configurations accepted in the configuration file or dictionary:
```json
{
  "model-cfg": "The path or filename of your yolov3.cfg",
  "model-weights": "The path or filename of your yolov3.weights",
  "model-classes": "The path of filename of your coco.names",
  "output-filename": "Suffix that is added to the output filename",
  "frame-skip": "[VIDEO] The number of frames to skip",
  "fps": "[VIDEO] FPS of the given video file",
  "confidence-threshold": "Minimum confidence value to keep the detected object",
  "nms-threshold": "Minimum non maximum supression value to keep the detected object in postprocessing",
  "input-width": "Width of network's input image",
  "input-height": "Height of network's input image"
  "show": "Show the video while its processed",
  "save": "Save the processed video",
  "gpu": "Use GPU to process videos",
}
```

Based from: 
https://github.com/spmallick/learnopencv/tree/master/ObjectDetection-YOLO