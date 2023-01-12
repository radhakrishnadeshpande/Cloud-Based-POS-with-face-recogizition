import cv2
# from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
# from matplotlib.patches import Rectangle
import warnings
warnings.simplefilter('ignore')
detector = MTCNN()
from numpy import asarray
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import boto3
import botocore
from PIL import Image
import pandas as pd
import io
import os
from twilio.rest import Client
import matplotlib
matplotlib.use('Agg')
from anvil import *
import anvil.server


anvil.server.connect("UW75BWDDDVWNNQTHAQXJZS5B-QD5Y4I6XTNMY7FVH")


@anvil.server.callable("capture")
def capture() :
	main()
	
def main():
	print("in uplink app")
	cam = cv2.VideoCapture(0)
	print(cam)
	count = 0
	while True:
		ret, img = cam.read()
		cv2.imshow("Test", img)
		if not ret:
			break
		k=cv2.waitKey(1)
		if k%256==27:
		#For Esc key
			print("Closed")
			break
		elif k%256==32: #For Space key
			print("Image"+str(count)+" saved")
			file='./img'+'.jpg'
			cv2.imwrite(file, img)
			count +=1
		cam.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	anvil.server.wait_forever()
