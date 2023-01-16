import os
from flask import Flask,render_template,request,redirect,url_for
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import numpy as np
import six.moves.urllib as urlib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
sys.path.append("..")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH = MODEL + '/frozen_inference_graph.pb'
PATH_LABEL = os.path.join('data','mscoco_label_map.pbtxt')
NUM_CLASSES = 90


 