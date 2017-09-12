# Human-Motion-Recognition
Human Motion Recognition using OneClass SVM (support vector machines)

# OneClass SVM

I propose a motion recognition method using OneClass SVM (Support Vector Machines). The videoclips in the database are composed of frames on which I applied a filter called "Recursive filter". So the dataset is composed of filtered images. Using principal component analysis (PCA) the feature of human motion is extracted and then the OneClass SVM classifier is employed to classify the motion pattern which is in our case "WALKING" because we are using one class.

# Recursive Filter

The method is based on recursive tracking and reducing noise in order to improve the tracking capability.

<p align="center">
  <img width="460" height="300" src="https://github.com/aoahmed/Human-Motion-Recognition/blob/master/dataset/train/weighted27.jpg?raw=true">
</p>
