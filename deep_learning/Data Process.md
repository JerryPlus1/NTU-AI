# Methodology

### Dataset Overview and Preprocess

In the field of medical image analysis, automated diagnosis of chest X-rays is of paramount importance. The dataset used in this study is the “Chest X-Ray Images (Pneumonia)” dataset, provided by Paul Timothy Mooney and available on Kaggle. This dataset is specifically designed for the task of pneumonia detection and classification, offering a substantial resource for training and evaluating medical image analysis models.

#### Dataset Description

The “Chest X-Ray Images (Pneumonia)” dataset comprises a total of 5,863 chest X-ray images, categorized into two main classes:

1. **Normal**: Images in this category show no signs of pneumonia and represent normal chest X-ray findings.
2. **Pneumonia**: Images in this category display signs of pneumonia. This category is further subdivided into:
    - **Bacterial Pneumonia**: X-ray images showing pneumonia caused by bacterial infections.
    - **Viral Pneumonia**: X-ray images showing pneumonia caused by viral infections.

#### Dataset Structure

The dataset is organized into three primary folders based on usage:

- **train**: Contains images used for training the model.
- **val**: Contains images used for validating the model's performance during training.
- **test**: Contains images used for testing the model's final performance.

Each primary folder is further divided into two subfolders:

- **NORMAL**: Contains images of normal chest X-rays.
- **PNEUMONIA**: Contains images of chest X-rays with pneumonia signs, further divided into bacterial and viral pneumonia.

#### Importance of the Dataset

This dataset is of significant value in the realm of automated medical diagnosis. By providing a diverse set of chest X-ray images, it offers crucial support for developing and evaluating deep learning models. Researchers can use these images to train efficient classification models that automatically detect and categorize pneumonia in chest X-rays. This capability is vital for enhancing diagnostic accuracy and efficiency, particularly in settings with limited medical resources. The dataset’s ability to aid in the rapid and accurate diagnosis of pneumonia underscores its clinical relevance and utility.

#### Data Preprocess
> 图像预处理的过程分为以下几步
> 1. 统一大小到 224 x 224 pixels
> 2. 计算训练数据的统计数据，对平均值和标准差进行归一化处理
> 3. 执行自定义直方图均衡化
> 4. 进行k折交叉验证
> 5. 提供三个通道的张量图像


Image preprocessing is a crucial step to ensure the effectiveness of model training and to improve prediction accuracy. Below is a detailed description of the image preprocessing steps:

1. **Resize to 224 x 224 pixels**  
   In the initial stage of image processing, all images are resized to a uniform dimension of 224 x 224 pixels. This step ensures that the input data has a consistent shape, allowing the model to accept and process the image data. Typically, image scaling techniques are used to achieve this, while maintaining the aspect ratio to minimize distortion.

2. **Normalize using mean and standard deviation calculated from training data**  
   Before performing normalization, we need to compute statistical data from the training dataset, including the mean and standard deviation for each channel. By normalizing each image (i.e., subtracting the mean and dividing by the standard deviation), we can scale the image data to a similar range. This helps to accelerate the model training process and improve convergence speed. Normalized data allows the model to better capture features and enhance generalization.

3. **Perform custom histogram equalization**  
   Custom histogram equalization is a technique used to enhance image contrast. By adjusting the brightness and contrast of the image, the distribution of grayscale values becomes more uniform, which improves the visibility of details. This step is particularly useful for images with low contrast and helps to increase the recognition capability of the subsequent model.

4. **Conduct k-fold cross-validation**  
   To evaluate the model's performance and prevent overfitting, we use k-fold cross-validation. The dataset is divided into k subsets, with k-1 subsets used for training and the remaining subset used for validation. This approach allows for a more reliable assessment of the model's generalization ability and helps in selecting the best model parameters.

5. **Provide tensor images with three channels**  
   The final processed images are provided as tensors with three channels, representing the red, green, and blue channels (RGB). This format meets the input requirements of most deep learning models, enabling the model to fully utilize the color information of the images for training and prediction.

Through these preprocessing steps, we effectively prepare the image data, ensuring it meets the model's input requirements and improving the model's training efficacy and prediction accuracy.