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
 