import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


matrix_train = None
for image in os.listdir('/home/ahmed/Desktop/dataset1/train'):
    imgraw = cv2.imread(os.path.join('/home/ahmed/Desktop/dataset1/train', image), 0)
    imgvector = imgraw.reshape(160*120)
    try:
        matrix_train = np.vstack((matrix_train, imgvector))
    except:
        matrix_train = imgvector

# PCA
print('matrix_train',np.shape(matrix_train))
X_t = PCA(2).fit_transform(matrix_train)

#plt.scatter(X_t[:,0], X_t[:,1], c='y')
#plt.show()


clf = svm.OneClassSVM(nu=0.001, kernel='rbf', gamma=0.00000001)
clf.fit(X_t)

# live

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
time.sleep(1)


count = 0
mean = 0
matrix_test = None
while (1):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = np.resize(gray,(1,160*120))
	fgmask = fgbg.apply(gray)
	retval,threshold = cv2.threshold(fgmask,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	kernel = np.ones((1,2),np.uint8)
	erosion = cv2.erode(fgmask,kernel,iterations = 1)
	erosion = erosion.reshape(160*120)
	font = cv2.FONT_HERSHEY_SIMPLEX
	print('mean',mean)
	if (mean==0):
		cv2.putText(frame,'Unoccupied',(30,70), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
	elif (mean>0):
		cv2.putText(frame,'Walking',(30,70), font, 1.5, (0,255,0), 2, cv2.LINE_AA)
	else:
		cv2.putText(frame,'Threat',(30,70), font, 1.5, (0,0,255), 2, cv2.LINE_AA)
	if(np.mean(erosion)>1) and (count<45):
		try:
			matrix_test = np.vstack((matrix_test,erosion))
			count+=1
		except:
			matrix_test=erosion
	elif (count>=45):
		Y_t = PCA(2).fit_transform(matrix_test)
		prediction = clf.predict(Y_t)
		mean = np.mean(prediction)
		print ('pred label', prediction)
		print ('pred label', mean)
		count=0
		matrix_test=None
	if (np.mean(erosion)<0.05) :
		mean=0
	cv2.imshow('frame',frame)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break


cap.release()
cv2.destroyAllWindows()

