# import the necessary packages
import decimal

from tempfile import mkdtemp
from werkzeug import serving
import os
import requests
import ssl
from werkzeug.utils import secure_filename

import random
import string
import json
import cv2
from uuid import uuid4
import sys
import random
import tensorflow as tf
import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io
import tensorflow as tf
import keras
from utils import load_model


from flask import jsonify
from flask import Flask
from flask import request
import traceback

from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import get_multi_model_bin


try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)




def load_my_model(model_path, weights_path):
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    graph = tf.get_default_graph()
    model = load_model(model_path)
    model.load_weights(weights_path)

    return model, graph


# define a predict function as an endpoint 
@app.route("/detect", methods=["POST"])
def detect():

    input_path = generate_random_filename(upload_directory,"jpg")
    output_path = generate_random_filename(upload_directory, "jpg")

    try:
        url = request.json["url"]

        download(url, input_path)
       
        results = []

        bgr_img = cv.imread(input_path)
        bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)

        with graph.as_default():
            preds = model.predict(rgb_img)
            
        prob = np.max(preds)
        class_id = np.argmax(preds)

        results.append({'label': class_names[class_id][0][0], 'score': '{:.4}'.format(prob)})        

        callback = json.dumps(results)

        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        clean_all([
            input_path
            ])

if __name__ == '__main__':
    global upload_directory
    global model, graph
    global img_width, img_height
    global class_names
    

    upload_directory = '/src/upload/'
    create_directory(upload_directory)

    model_path = "/src/models/"
    model_file = "resnet152_weights_tf.h5"
    weights_file = "model.96-0.89.hdf5"

    cloud_storage_prefix = "https://storage.gra5.cloud.ovh.net/v1/AUTH_18b62333a540498882ff446ab602528b/pretrained-models/image/car-classifier/v0/"

    get_multi_model_bin([
        (cloud_storage_prefix + model_file, model_path + model_file),
        (cloud_storage_prefix + weights_file, model_path + weights_file)
        ])

    model, graph = load_my_model(model_path + model_file, model_path + weights_file)


    img_width, img_height = 224, 224

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)

