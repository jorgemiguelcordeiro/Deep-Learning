



# **1 - Introduction & Task Definition**  
This project develops a Deep Learning model to classify mask usage based on images of people. The objective is to automate the detection of correct mask placement using a model that, when combined with a camera, can determine whether individuals are wearing masks properly before granting access to restricted areas.  



# **2 - Data Description**  
The dataset consists of four categories:  

- **Fully Covered** – Faces with masks covering the nose and mouth.  
- **Partially Covered** – Faces with masks covering only the mouth.  
- **Not Covered** – Faces without masks.  
- **Not a Face** – Images detected as faces by OpenCV but not containing actual faces.  

The dataset was sourced from Kaggle ([link](https://www.kaggle.com/datasets/jamesnogra/face-mask-usage)) and collected using OpenCV. The images originate from YouTube videos and mobile recordings. The model is trained to recognize masks under different conditions, including occlusion, deformations, and various lighting environments.  



# **3 - Evaluation Measures**  
Standard metrics for multi-class classification problems were employed, including **Accuracy, Recall, and F1-Score**. Additionally, special attention was given to the **Precision of the Fully Covered class**, as false positives in this category represent the most critical misclassification. To prevent unnecessary computation and overfitting, an **Early Stopping** callback was implemented, halting training if validation accuracy did not improve for two consecutive epochs.  

The initial strategy involved training models using the **Hold-out Method** to identify promising parameters in terms of accuracy, precision, and training time. Once the best configurations were identified, **Cross-Validation** was applied to obtain more realistic metric values for the selected models.  



# **4 - Hold-out Method**  
The first approach utilized was the **Hold-out Method**, where the dataset was sequentially split into:  

- **80% for training**  
- **10% for validation**  
- **10% for testing**  

To automate this process, Python functions were implemented to create structured folders for train, validation, and test sets, with subfolders corresponding to each of the four classes. The images were rescaled from **0-255 to 0-1** to improve model efficiency.  

Additionally, **data generators** were developed to yield batches of **64×64px RGB images** with various batch sizes. For example, with an image size of **64×64px** and batch size of **20**, the generator produces images with shape **(20, 64, 64, 3)** and binary labels with shape **(20, 4)**.  

To streamline evaluation, **ten functions** were designed to test different combinations of image and batch sizes.  

Five distinct architectures were tested, iteratively adjusting hyperparameters, including **image size, batch size, dropout rate, and convolutional layers**. The first four models shared a similar structure, consisting of:  

1. Multiple **convolutional layers**, each followed by a **max pooling layer**.  
2. A **flatten layer** after the convolutional layers.  
3. **Fully connected (dense) layers**, with the last layer containing **four neurons** and a **softmax activation function** for multi-class classification.  


# **5 - Cross-Validation**  
Following the **Hold-out Method**, the **Stratified K-Fold Cross-Validation** technique was implemented to improve evaluation reliability. This approach involves splitting the training data into **K subsets**, ensuring each subset maintains the class distribution. The model is trained on **K-1 subsets**, while the remaining subset is used for validation. This process is repeated until every subset has been used for evaluation, and the final results are averaged.  

The key advantage of **Cross-Validation** over the **Hold-out Method** is its ability to mitigate the effects of **biases or anomalies** in the dataset, leading to more stable performance metrics.  

For this study, after selecting the **top 5 models** from the Hold-out Method (with **64×64px images** and batch sizes ranging from **20 to 64**), **5-Fold Cross-Validation** (**K = 5**) was applied. Each iteration was limited to **15 epochs**, as models previously exhibited overfitting tendencies beyond that point.  

However, a significant issue arose when applying **Cross-Validation** to the **AlexNet architecture (Model 5)**, as it consistently caused system crashes due to high computational requirements. This highlights a fundamental limitation of **Cross-Validation**—its significantly higher processing power demands. Given the available computational resources, this proved to be a **notable drawback** of the method.  

# 6 - Results obtained
In the table below, in the row "Architecture", the
numbers without letters refer to the number of neurons in the convolutional layers (each one being
followed by a max pooling layer); "f" refers to a flatten layer; "d(x)" refers to a dropout layer, with value
x; and “D(y)” refers to a dense layer, with value y
![image](https://github.com/user-attachments/assets/1efe8f8d-133f-4607-899e-c65f4f208034)

