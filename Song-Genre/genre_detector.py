# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:16:42 2020

@author: Admin
"""

import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

y, _ = librosa.load("D:\\Machine Learning_Algoritms\\Song-Genre\\kick_loop.wav")
mfcc = librosa.feature.mfcc(y)
plt.figure(figsize=(10,4))
librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
plt.colorbar()
plt.title("song")
plt.tight_layout()
plt.show()