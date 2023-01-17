import cv2
from keras.models import load_model
from PIL import Image
import os
import numpy as np

model=load_model('Braintumor1st2nd.h5')
image=cv2.imread('C:\\Project clg\\pred\\pred0.jpg')
img=Image.fromarray(image)
img=img.resize((64,64))

img = np.array(img)
print(img)
#if all as zero no cancer or it is tumor



