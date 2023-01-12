# from matplotlib import pyplot as plt
# from mtcnn.mtcnn import MTCNN
# import warnings
# from matplotlib.patches import Rectangle
# warnings.simplefilter('ignore')
# detector = MTCNN()
# def highlight_faces(image_path, faces):
# 	image = plt.imread(image_path)
# 	plt.imshow(image)
# 	ax = plt.gca()
# 	for face in faces:
# 		x, y, width, height = face['box']
# 		face_border = Rectangle((x, y), width, height,fill=False, color='red')
# 		ax.add_patch(face_border)
# 	plt.show()

# image = plt.imread('img1.jpg')
# faces = detector.detect_faces(image)
# for face in faces:
# 	print(face)

# highlight_faces('img1.jpg', faces)

# from numpy import asarray
# from PIL import Image
# import warnings
# warnings.filterwarnings('ignore')

# def extract_face_from_image(image_path, required_size=(224, 224)):
# 	image = plt.imread(image_path)
# 	detector = MTCNN()
# 	faces = detector.detect_faces(image)
# 	face_images = []
# 	for face in faces:
# 		x1, y1, width, height = face['box']
# 		x2, y2 = x1 + width, y1 + height
# 		face_boundary = image[y1:y2, x1:x2]
# 		face_image = Image.fromarray(face_boundary)
# 		face_image = face_image.resize(required_size)
# 		face_array = asarray(face_image)
# 		face_images.append(face_array)
# 	return face_images
# extracted_face = extract_face_from_image('img1.jpg')
# # Display the first face from the extracted faces
# plt.imshow(extracted_face[0])
# plt.show()

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
dataset=datasets.ImageFolder(os.getcwd() +'/images')
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}
def collate_fn(x):
	return x[0]
loader = DataLoader(dataset, collate_fn=collate_fn)
# print(loader)
face_list = []
name_list = []
embedding_list = []
for img, idx in loader:
	print(img)
	face, prob = mtcnn(img, return_prob=True)
	if face is not None and prob>0.90: # if face detected and porbability > 90%
		emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
		embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
		name_list.append(idx_to_class[idx]) # names are stored in a list
# print(embedding_list)
data = [embedding_list, name_list]
torch.save(data, 'data.pt')


def face_match(img_path, data_path): # 
	img = Image.open(img_path)
	face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
	emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
	saved_data = torch.load('data.pt') 
	embedding_list = saved_data[0] # getting embedding data
	name_list = saved_data[1] # getting list of names
	dist_list = [] # list of matched distances, minimum distance is used to identify the person
	for idx, emb_db in enumerate(embedding_list):
		dist = torch.dist(emb, emb_db).item()
		dist_list.append(dist)
		idx_min = dist_list.index(min(dist_list))
	return (name_list[idx_min], min(dist_list))



result = face_match('img0.jpg', 'data.pt')
print(result)
print('Face matched with: ',result[0], 'With distance: ',result[1])
