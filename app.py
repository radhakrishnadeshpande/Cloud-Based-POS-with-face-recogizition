from flask import Flask, render_template,jsonify,request
app = Flask(__name__)
from PIL import Image
from io import BytesIO
import base64
import cv2
# from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
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

from threading import Thread

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def face_match(img_path, data_path):
# img_path= location of photo, data_path= location of data.pt
#getting embedding matrix of the given img
   img = Image.open(img_path).convert('RGB')
   mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
   resnet = InceptionResnetV1(pretrained='vggface2').eval()
   face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability

   try:
      emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
      saved_data = torch.load('data.pt') # loading data.pt file
      embedding_list = saved_data[0] # getting embedding data
      name_list = saved_data[1] # getting list of names
      dist_list = [] # list of matched distances, minimum distance is used to identify the person
      for idx, emb_db in enumerate(embedding_list):
         dist = torch.dist(emb, emb_db).item()
         dist_list.append(dist)
         idx_min = dist_list.index(min(dist_list))
      # print(dist_list)
      return (name_list[idx_min], min(dist_list))
   except Exception as e:
      # raise e
      return None

def run_script():
    file = open(r'/path/to/your/training.py', 'r').read()
    return exec(file)

def run_func():
    data = { 'some': 'data', 'any': 'data' }
    thr = Thread(target=run_script, args=[app, data])
    thr.start()
    return thr

@app.route("/")
def index():
   return render_template("title.html")
   
@app.route('/password')
def password():
   return render_template('password.html')

@app.route('/new_user')
def new_user():
   return render_template('register.html')

@app.route('/capture_image')
def capture_image():
   return render_template('index.html')





@app.route('/verify_image',methods=['POST'])
def verify_image():
   # print(request.form)
   # data=request.form.to_dict()
   image_data=request.form.get('filename').split(",")[1]
   # print(image_data)
   # image_data = bytes(image_data, encoding="ascii")
   im = Image.open(BytesIO(base64.b64decode(image_data)))
   im.save('image.png')
   response=face_match('image.png',"data.pt")
   print(response)
   if response==None:
      return render_template('faceVerification.html')
   else:
      return render_template('enterpass.html',res={"user":response})
   return jsonify(response)




#register
@app.route('/register',methods=['POST'])
def register():
   image_data=request.form.get('filename').split(",")[1]
   # print(image_data)
   user=request.form.get('username')
   # image_data = bytes(image_data, encoding="ascii")
   im = Image.open(BytesIO(base64.b64decode(image_data)))
   os.makedirs('./images/'+user)
   im.save('./images/'+user+'/image.png')
   # run_func()
   return render_template('RegisterSuccess.html')
#password 


@app.route('/verify_pass',methods=['POST'])
def verify_pass():
   data=request.form.to_dict()
   user=data['user']
   pwd=data['psw']
   if user=='Radha Krishna'and  pwd=='0801':
      return render_template('payment.html')
   else:
      return render_template('verificationfailed.html')
   return jsonify(request.form.to_dict())

if __name__ == '__main__':
   app.run(debug = True)




   