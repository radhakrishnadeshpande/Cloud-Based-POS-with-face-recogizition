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

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

dataset=datasets.ImageFolder('./images')
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}
def collate_fn(x):
	return x[0]
loader = DataLoader(dataset, collate_fn=collate_fn)
face_list = []
name_list = []
embedding_list = []
for img, idx in loader:
	face, prob = mtcnn(img, return_prob=True)
	if face is not None and prob>0.90: # if face detected and porbability > 90%
		emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to getembedding matrix
		embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
		name_list.append(idx_to_class[idx]) # names are stored in a list
# print(embeddin_list)
data = [embedding_list, name_list]
torch.save(data, 'data.pt')
# return True