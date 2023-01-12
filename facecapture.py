import cv2
cam = cv2.VideoCapture(0)
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
	elif k%256==32:
	#For Space key
		print("Image"+str(count)+" saved")
		file='img'+str(count)+'.jpg'
		cv2.imwrite(file, img)
		count +=1
cam.release()
cv2.destroyAllWindows()


