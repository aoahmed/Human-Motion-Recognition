# Human-Motion-Recognition
Human Motion Recognition using OneClass SVM (support vector machines)

# OneClass SVM

I propose a motion recognition method using OneClass SVM (Support Vector Machines). The videoclips in the database are composed of frames on which I applied a filter called "Recursive filter". So the dataset is composed of filtered images. Using principal component analysis (PCA) the feature of human motion is extracted and then the OneClass SVM classifier is employed to classify the motion pattern which is in our case "WALKING" because we are using one class.

# Recursive Filter

The method is based on recursive tracking and reducing noise in order to improve the tracking capability.

<p align="center">
  <img width="400" height="300" src="https://github.com/aoahmed/Human-Motion-Recognition/blob/master/dataset/train/weighted27.jpg?raw=true">
</p>

# Preparing dataset

```
matrix_train = None
for image in os.listdir('/home/ahmed/Desktop/dataset1/train'):
    imgraw = cv2.imread(os.path.join('/home/ahmed/Desktop/dataset1/train', image), 0)
    imgvector = imgraw.reshape(160*120)
    try:
        matrix_train = np.vstack((matrix_train, imgvector))
    except:
        matrix_train = imgvector
```
# Results !!!!

There are 3 status : "UNOCCUPIED" , "WALKING" and "THREAT" if the new data is different from the training set (which is walking)

<p align="center">
  <img width="200" height="200" src="https://github.com/aoahmed/Human-Motion-Recognition/blob/master/dataset/unoccupied.PNG?raw=true">
  <img width="200" height="200" src="https://github.com/aoahmed/Human-Motion-Recognition/blob/master/dataset/walking.PNG?raw=true">
  <img width="200" height="200" src="https://github.com/aoahmed/Human-Motion-Recognition/blob/master/dataset/threat.PNG?raw=true">
</p>

This is the  classification scheme where we can see the training (yellow dots) and testing (red dots) sets retain the same proportion and the the accuracy level of prediction is about 0.899/1.
<p align="center">
  <img width="400" height="300" src="https://github.com/aoahmed/Human-Motion-Recognition/blob/master/dataset/class.PNG?raw=true">
</p>

# Improvements !!!

As an Improvement, I have to add more classes to recognize more motions and push the accuracy more and more to 98%.
