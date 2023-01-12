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


from anvil import *
import anvil.server


# anvil.server.connect("UW75BWDDDVWNNQTHAQXJZS5B-QD5Y4I6XTNMY7FVH")

# cam = cv2.VideoCapture(0)
# count = 0
# while True:
# 	ret, img = cam.read()
# 	cv2.imshow("Test", img)
# 	if not ret:
# 		break
# 	k=cv2.waitKey(1)
# 	if k%256==27:
# 	#For Esc key
# 		print("Closed")
# 		break
# 	elif k%256==32:
# 	#For Space key
# 		print("Image"+str(count)+" saved")
# 		file='E:/POS/new/img'+'.jpg'
# 		cv2.imwrite(file, img)
# 		count +=1
# 		cam.release()
# 		cv2.destroyAllWindows()
# #print bounding box of the face


# image = plt.imread('./img.jpg')

# faces = detector.detect_faces(image)
# for face in faces:
# 	print(face)
#Display image with bounding box.

# def highlight_faces(image_path, faces):
# 	# display image
# 	image = plt.imread(image_path)
# 	plt.imshow(image)
# 	ax = plt.gca()
# 	# for each face, draw a rectangle based on coordinates
# 	for face in faces:
# 		x, y, width, height = face['box']
# 		face_border = Rectangle((x, y), width, height,fill=False, color='red')
# 		ax.add_patch(face_border)
# 	plt.show()
# 	plt.close()
# highlight_faces('E:/POS/new/img.jpg', faces)
# #Extract the face and display
# def extract_face_from_image(image_path, required_size=(224, 224)):
# # load image and detect faces
# 	image = plt.imread(image_path)
# 	detector = MTCNN()
# 	faces = detector.detect_faces(image)
# 	face_images = []
# 	# extract the bounding box from the requested face
# 	for face in faces:
# 		x1, y1, width, height = face['box']
# 		x2, y2 = x1 + width, y1 + height
# 		# extract the face
# 		face_boundary = image[y1:y2, x1:x2]
# 		# resize pixels to the model size
# 		face_image = Image.fromarray(face_boundary)
# 		face_image = face_image.resize(required_size)
# 		face_array = asarray(face_image)
# 		face_images.append(face_array)
# 	return face_images
# extracted_face = extract_face_from_image('E:/POS/new/img.jpg')
# # Display the first face from the extracted faces
# plt.imshow(extracted_face[0])
# plt.show()
# #create embedding list and verify face
# importing libraries

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
#
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
# def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt
# 	# getting embedding matrix of the given img
# 	img = Image.open(img_path)
# 	face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
# 	emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
# 	saved_data = torch.load('data.pt') # loading data.pt file
# 	embedding_list = saved_data[0] # getting embedding data
# 	name_list = saved_data[1] # getting list of names
# 	dist_list = [] # list of matched distances, minimum distance is used to identify the person
# 	for idx, emb_db in enumerate(embedding_list):
# 		dist = torch.dist(emb, emb_db).item()
# 		dist_list.append(dist)
# 		idx_min = dist_list.index(min(dist_list))
# 	return (name_list[idx_min], min(dist_list))
# # result = face_match('E:/POS/new/img.jpg', 'data.pt')
# # print('Face matched with: ',result[0], 'With distance: ',result[1])




# @anvil.server.callable("capture")
# def capture() :
# 	print("in uplink app")
# 	cam = cv2.VideoCapture(0)
# 	count = 0
# 	while True:
# 		ret, img = cam.read()
# 		cv2.imshow("Test", img)
# 		if not ret:
# 			break
# 		k=cv2.waitKey(1)
# 		if k%256==27:
# 		#For Esc key
# 			print("Closed")
# 			break
# 		elif k%256==32: #For Space key
# 			print("Image"+str(count)+" saved")
# 			file='./img'+'.jpg'
# 			cv2.imwrite(file, img)
# 			count +=1
# 		cam.release()
# 		cv2.destroyAllWindows()

# #Uploading Image
# @anvil.server.callable("uploadImage")
# def uploadImage(customerId):
# 	file_name = './img.jpg'
# 	bucket = 'cloudbasedposcustomerimages'
# 	key = 'images/'+customerId+'/'+customerId
# 	client = boto3.client('s3', aws_access_key_id="",\
#                              aws_secret_access_key="", region_name="")
# 	client.upload_file(file_name, bucket, key )#stores the image in the cloud
# #Uploading Password
# @anvil.server.callable("uploadPassword")
# def uploadPassword(password,customerId):
# 	client = boto3.client('s3', aws_access_key_id="",\
#                              aws_secret_access_key="", region_name="")
# 	pwdbucket='cloudbasedposcustomerpasswords'
# 	pkey='Passwords/'+customerId
# 	client.put_object(Body=password,Bucket=pwdbucket,Key=pkey)#stores password in cloud
# #verify Image
# def face_match(img_path, data_path):
# # img_path= location of photo, data_path= location of data.pt
# #getting embedding matrix of the given img
# 	img = Image.open(img_path)
# 	mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
# 	resnet = InceptionResnetV1(pretrained='vggface2').eval()
# 	face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
# 	emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
# 	saved_data = torch.load('data.pt') # loading data.pt file
# 	embedding_list = saved_data[0] # getting embedding data
# 	name_list = saved_data[1] # getting list of names
# 	dist_list = [] # list of matched distances, minimum distance is used to identify the person
# 	for idx, emb_db in enumerate(embedding_list):
# 		dist = torch.dist(emb, emb_db).item()
# 		dist_list.append(dist)
# 		idx_min = dist_list.index(min(dist_list))
# 	return (name_list[idx_min], min(dist_list))

# #verify customer Image
# @anvil.server.callable("verifyCustomerImage")
# def verifyCustomerImage() :
# 	BUCKET_NAME = 'cloudbasedposcustomerimages'
# 	s3 = boto3.resource('s3')
# 	my_bucket = s3.Bucket(BUCKET_NAME)
# 	for file in my_bucket.objects.all():
# 		KEY = file.key
# 		print(KEY)
# 	try:
# 		s3.Bucket(BUCKET_NAME).download_file(KEY, './img.jpg')
# 		mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
# 		resnet = InceptionResnetV1(pretrained='vggface2').eval()
# 		dataset=datasets.ImageFolder('./images')
# 		idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}
# 		def collate_fn(x):
# 			return x[0]
# 		loader = DataLoader(dataset, collate_fn=collate_fn)
# 		face_list = []
# 		name_list = []
# 		embedding_list = []
# 		for img, idx in loader:
# 			face, prob = mtcnn(img, return_prob=True)
# 			if face is not None and prob>0.90: # if face detected and porbability > 90%
# 				emb = resnet(face.unsqueeze(0))
# 				embedding_list.append(emb.detach())
# 				name_list.append(idx_to_class[idx]) # names are stored in a list
# 		data = [embedding_list, name_list]
# 		torch.save(data, 'data.pt')
# 		result = face_match('./img.jpg', 'data.pt')
# 		os.remove('./imgjpg')
# 		if (result[1] < 0.9999):
# 			removeExtensionFromKey = KEY.split(".")[0]
# 			customerId = removeExtensionFromKey.split("/")[2]
# 		return customerId,True
# 	except botocore.exceptions.ClientError as e:
# 		if e.response['Error']['Code'] == "404":
# 			print("The object does not exist.")
# 		else:
# 			raise
# 		return 0,False
# #verify password



# @anvil.server.callable("verifyPassword")
# def verifyPassword(password,customerId) :
# 	newKey = 'Passwords' +'/'+ customerId
# 	s3_client = boto3.client('s3', use_ssl=False)
# 	bucket = 'cloudbasedposcustomerpasswords'
# 	obj = s3_client.get_object(Bucket=bucket, Key=newKey)
# 	passwordInS3=pd.read_fwf((io.BytesIO(obj['Body'].read())) , encoding= 'unicode_escape',
# 	delimiter='|', error_bad_lines=False,header=None, dtype=str)
# 	print(passwordInS3[0][0])
# 	if passwordInS3[0][0] == password :
# 		print("Passwords are matched and complete payment")
# 		return True
# 	else :
# 		return False
# #send message to customer
# def sendClientMessage(phoneNo,remainedAmount) :
# 	# importing twilio
# 	# Your Account Sid and Auth Token from twilio.com / console
# 	account_sid = 'ACa2b6861be817deaf836be05c02952410'
# 	auth_token = '972b861ae5145bddba07978b135a65b5'
# 	client = Client(account_sid, auth_token)
# 	body = 'Remained Amount is '+str(remainedAmount)
# 	message = client.messages.create(
# 	from_='+19203153332',
# 	body = body,
# 	to = phoneNo
# 	)
# #Processing Payment
# @anvil.server.callable("processPayment")
# def processPayment(accountId,amount,phoneNo) :
# 	s3 = boto3.resource('s3')
# 	s3_client = boto3.client('s3', use_ssl=False)
# 	bucket = s3.Bucket('cloudbasedposaccountno')
# 	obj=s3_client.get_object(Bucket='cloudbasedposaccountno',Key='AccountNo/'+accountId)
# 	amountInBank = pd.read_fwf((io.BytesIO(obj['Body'].read())),encoding= 'unicode_escape',delimiter='|', error_bad_lines=False,header=None, dtype=str)
# 	print("Amount in the bank account is ",amountInBank[0][0])
# 	if int(amountInBank[0][0]) >= int(amount):
# 		remainedAmount = (int(amountInBank[0][0]) - int(amount))
# 		print("Amount deducted now is ",remainedAmount)
# 		s3_client.put_object(Body=str(remainedAmount),Bucket='cloudbasedposaccountno',Key='AccountNo/'+accountId)
# 		sendClientMessage(phoneNo, remainedAmount)
# 		return True
# 	else :
# 		return False
# #Customer Type Module



# anvil.server.wait_forever()

# # class BootingPage(BootingPageTemplate):
# # 	def init (self, **properties):
# # 		self.init_components(**properties)
# # 	def button_1_click(self, **event_args):
# # 	#anvil.server.call("newUser",self.button_1.text)
# # 		open_form('NewUser')
# # 		pass
# # 	def label_1_show(self, **event_args):	
# # 		"""This method is called when the Label is shown on the screen"""
# # 		pass
# # 	def button_2_click(self, **event_args):
# # 		"""This method is called when the button is clicked"""
# # 		open_form('ExistingUser')
# # 		pass
# # #Existing user module

# # class ExistingUser(ExistingUserTemplate):
# # 	def init (self, customerId=0):
# # 		self.ExistingUser_set_customerId(customerId)
# # 	# getter method
	
# # 	def ExistingUser_get_Id(self):
# # 		return self._customerId
# # 	# setter method
	
# # 	def ExistingUser_set_customerId(self, x):
# # 		self._customerId = x
	
# # 	def button_1_click(self, **event_args):
# # 		"""This method is called when the button is clicked"""
# # 		anvil.server.call("capture")
# # 		self.button_2.visible = True
# # 		pass
	
# # 	def button_2_click(self, **event_args):
# # 		customerId,faceVerified = anvil.server.call("verifyCustomerImage")
# # 		self.ExistingUser_set_customerId(customerId)
# # 		if (faceVerified) :
# # 			self.button_3.visible = True
# # 		else :
# # 			open_form('FaceVerificationFailed')
# # 			pass
	
# # 	def button_3_click(self, **event_args):
# # 		"""This method is called when the button is clicked"""
# # 		self.text_box_1.visible = True
# # 		pass
	
# # 	def text_box_1_pressed_enter(self, **event_args):
# # 		"""This method is called when the user presses Enter in this text box"""
# # 		customerId = self.ExistingUser_get_Id()
# # 		#print(customerId)
# # 		passwordVerified = anvil.server.call("verifyPassword",self.text_box_1.text,customerId)
# # 		if (passwordVerified) :
# # 			open_form('PaymentProcessor')
# # 			return
# # 		else :
# # 			open_form('PasswordVerificationFailed')
# # 		#print(self.ExistingUser_get_Id())
# # 			pass


# # class FaceVerificationFailed(FaceVerificationFailedTemplate):
# # 	def init (self, **properties):
# # 		# Set Form properties and Data Bindings.
# # 		self.init_components(**properties)
# # #New User Module



# # class NewUser(NewUserTemplate):
# # 	def init (self, **properties):
# # 	# Set Form properties and Data Bindings.
# # 		self.init_components(**properties)
# # 	def button_1_click(self, **event_args):
# # 		"""This method is called when the button is clicked"""
# # 		anvil.server.call("capture")
# # 		self.button_2.visible = True
# # 		pass
# # 	def text_box_1_pressed_enter(self, **event_args):
# # 		anvil.server.call("uploadImage",self.text_box_1.text)
# # 		self.button_3.visible = True
# # 		pass
# # 	def text_box_1_hide(self, **event_args):
# # 		pass
# # 	def button_2_click(self, **event_args):
# # 		self.text_box_1.visible = True
# # 		pass
# # 	def button_3_click(self, **event_args):
# # 		self.text_box_2.visible = True
# # 		pass
# # 	def text_box_2_pressed_enter(self, **event_args):
# # 		anvil.server.call("uploadPassword",self.text_box_2.text,self.text_box_1.text)
# # 		open_form('SucessfullyRegistered')
# # 		pass




# # class PasswordVerificationFailed(PasswordVerificationFailedTemplate):
# # 	def init (self, **properties):
# # 	# Set Form properties and Data Bindings.
# # 		self.init_components(**properties)



# # #Payment failed page

# # class PaymentFailed(PaymentFailedTemplate):
# # 	def init (self, **properties):
# # 		self.init_components(**properties)

# # #Payment processing module


# # class PaymentProcessor(PaymentProcessorTemplate):
# # 	def init (self, **properties):
# # 		self.init_components(**properties)
# # 	def text_box_1_pressed_enter(self, **event_args):
# # 		pass
# # 	def text_box_2_pressed_enter(self, **event_args):
# # 		pass
# # 	def button_1_click(self, **event_args):
# # 		accountId = self.text_box_1.text
# # 		amount = self.text_box_2.text
# # 		phoneNo = self.text_box_3.text
# # 		x=anvil.server.call("processPayment",accountId,amount,phoneNo)
# # 		if(x):
# # 			open_form('ThankYou')
# # 		else :
# # 			open_form('PaymentFailed')
# # 			pass
# # 	def text_box_3_pressed_enter(self, **event_args):
# # 		pass

# # #Successfully registered page
# # class SucessfullyRegistered(SucessfullyRegisteredTemplate):
# # 	def init (self, **properties):
# # 		self.init_components(**properties)
# # #Thankyou Page

# # class ThankYou(ThankYouTemplate):
# # 	def init (self, **properties):
# # 		self.init_components(**properties)