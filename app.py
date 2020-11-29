from IPython.display import clear_output, Image, display
import PIL.Image as image
import io
import json
import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import wget
import pickle
import os
from flask import Flask, request, jsonify, render_template

from predict import lxmert
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        #self.question = 
        self.answer = lxmert(self.filename)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predictRoute():
    image = request.json['image']
    
    result = clApp.answer.prediction()
    return jsonify(result)









if __name__ == '__main__':
    clApp=ClientApp()
    app.run(host='localhost', port=8000, debug=True)
