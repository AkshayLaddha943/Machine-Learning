# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:56:57 2020

@author: Admin
"""

import cv2
from flask import Flask, request, make_response
import numpy as np
import urllib.request

app = Flask(__name__)

@app.route("/canny", methods=['GET'])
def canny_process():
    with urllib.request.urlopen(request.args.get('url')) as url:
        img_array = np.asarray(bytearray(url.read()), dtype=np.uint8)
    
    #Convert image to opencv format    
    img_cv = cv2.imdecode(img_array, -1)
    
    #Convert image to grayscale
    img_grey = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    #performing canny edge detection
    edges = cv2.Canny(img_grey, 100, 300)
    
    #Compress image and store it in the memory buffer
    retval, buffer = cv2.imencode(".jpg", edges)
    
    #Build the response:
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'

    # Return the response:
    return response

if __name__=="__main__":
    app.run(host='0.0.0.0', debug=False)