# 1 - Introduction & Task Definition
This project develops a Deep Learning model to classify mask usage based on images of people. The goal is to automate the detection of correct mask placement using a model that, when combined with a camera, can determine whether individuals are wearing masks properly before granting access to restricted areas.

# 2 - Data Description
The dataset consists of four categories:

 - Fully Covered – Faces with masks covering nose and mouth.
 - Partially Covered – Faces with masks covering only the mouth.
 - Not Covered – Faces without masks.
 - Not a Face – Images detected as faces by OpenCV but not containing actual faces.
   
The dataset was sourced from Kaggle (https://www.kaggle.com/datasets/jamesnogra/face-mask-usage) and collected using OpenCV, featuring images from YouTube videos and mobile recordings. The model is trained to recognize masks under different conditions, including occlusion, deformations, and various lighting environments.

# 3 - Evaluation Measures
it's used the usual metrics in Multi-Class Classification Problems such as Accuracy,
Recall and F1-Score. Precision of the Fully Covered class was also a measure we looked at
closely to evaluate the models as we believe that False Positives in this class are the worse prediction
mistakes the model can make and thus it needs to be kept to a minimum. With all this in mind, our strategy was to initially use the Hold-out Method to train the models
and see which parameters are more promising in terms of accuracy, precision and time taken to train.
Then, these more promising parametrizations are selected use the Cross-Validation method, in order
to get a more realistic value of the metrics for the selected models.

# 4 - Holdout method
The first approach was to use the Hold-out method. We decided to save the images
sequentially, assigning the first 80% of data of each image class to the train set, the following 10% to
the validation set and the last 10% to the test set. Instead of doing this process manually, we used
Python functions to automate it, thus creating folders for train, validation, and test sets, as well as four
subfolders for each set representing each one of the four classes the dataset has. Consequently,
each image will be assigned to the folder it belongs to.
The pixels of the images of the dataset contained values that ranged from 0 to 255, so we had
to perform a rescaling of all the images to fit the range from 0 to 1. Knowing that we would have to
test our models with different image and batch sizes, we created generators that yield batches of RGB
images of different sizes and binary labels (for example, for image size 64x64px with batch size 20,
the generator yields batches of 64x64px RGB images with shape (20,64,64,3) and binary labels with
shape (20)).
We also created, from the get-go, ten functions that received a model and a directory to
perform the model evaluation (one for each combination of image and batch size that we wanted to
test).
After this step, we decided to try five distinct architectures, with different hyperparameters.
With it, we proceeded to train them in an interactive way, changing the hyperparameters along the
way, in order to find the right configuration for our goal. Being so, we tried to alter the values of the
image and batch size, and some of the models' layers - as the quantity and the value in the dropout
and convolutional layers.
The first 4 models we tested have a similar structure. Thus, they are composed of several
convolutional layers, each one being followed by a max pooling layer; after that, we have a flatten
4
layer; following, we resort to dense layers (the last one has 4 neurons and is the only layer of our
models whose activation function is not relu, since we want a prediction to be made concerning the 4
classes, having therefore chosen to resort to softmax)

# 5 - Cross - validation
After using the hold-out method to train and evaluate our models, we decided to implement
the Stratified K-Fold Cross-Validation method, which consists in splitting the available train dataset
into K subsets (within each subset, the percentage of samples of each target class is roughly the
same), choosing one of the subsets to be used for evaluation, and using the other K-1 subsets to train
the model. This process is then repeated until all the subsets have been used to evaluate the model.
Finally, the results of every iteration are then combined to get the average model evaluation. One
advantage over the hold-out method is that the metrics provided by this approach generally present
a more realistic value of the selected models since it helps to minimize the effect of strange or
undesirable patterns present on the dataset by splitting it into subsets and then averaging the metrics.
In our case, after choosing the best 5 models with the hold-out method concerning parameters
(64x64px for the image size, with the exception for models 4 and 5; and batch sizes of 20 for model
1, 2 and 4; 32 for model 3; and 64 for model 5), we decided to test them with this new method to
observe if any changes in the models’ scores would happen. However, because we realised that this
method would be quite expensive in terms of time, we opted for batch sizes of 64 right from the start.
To start, we had to convert all the images in our dataset into a NumPy array, so that we could
use scikit-learn train_test_split. The split consisted of saving 10% of our dataset to be used for testing
purposes later on in the final models. We decided on using 5 folds (K = 5) for all the models, since a
lot of our research shows us that it is a good value for K. We let each iteration run for 15 epochs,
since almost all the runs done previously (on the hold-out method) stopped or started to overfit after
that. One problem came up when trying to run this method on the AlexNet architecture (model 5),
5
since it would always crash our laptops every time we ran it. Because of this, we opted to not run this
model on this part of the project. This is one very big disadvantage brought by cross-validation
methods, since they require much more processing power and since we are limited by the resources
we have, we felt like this can be a negative point for cross-validation in general.
