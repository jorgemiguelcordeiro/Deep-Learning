# Introduction & Task Definition
This project develops a Deep Learning model to classify mask usage based on images of people. The goal is to automate the detection of correct mask placement using a model that, when combined with a camera, can determine whether individuals are wearing masks properly before granting access to restricted areas.

# Data Description
The dataset consists of four categories:

 - Fully Covered – Faces with masks covering nose and mouth.
 - Partially Covered – Faces with masks covering only the mouth.
 - Not Covered – Faces without masks.
 - Not a Face – Images detected as faces by OpenCV but not containing actual faces.
   
The dataset was sourced from Kaggle (https://www.kaggle.com/datasets/jamesnogra/face-mask-usage) and collected using OpenCV, featuring images from YouTube videos and mobile recordings. The model is trained to recognize masks under different conditions, including occlusion, deformations, and various lighting environments.


