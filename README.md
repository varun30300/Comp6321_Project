# Concordia University

Department of Computer Science and Software Engineering COMP 6321

MACHINE LEARNING

Project Group M

## Authors

Name: Saumya Shah <br/>
Applicant ID: 40167279

Name: Vikyath Srinivasulu <br/>
Applicant ID: 40218245

Name: Varun Pandey <br/>
Applicant ID: 40225320

Name: Unnati Chaturvedi <br/>
Applicant ID: 40227872

## High Level Description of Project
This project investigates image classification techniques through the analysis of three distinct datasets: medical diagnostics (highlighting colorectal and prostate cancer) and animal face recognition. Our focus is on assessing the practical application and efficiency of Convolutional Neural Networks (CNNs) in classifying real-world images.

In the first task, we engage in training a CNN to identify colorectal cancer. This process involves meticulous hyperparameter tuning along with the utilization of pre-trained network models. The outcomes are visualized and interpreted using t-SNE analysis. The second task shifts focus to the extraction of features from new datasets (prostate cancer and animal faces) using pre-trained CNN encoders. This allows for a comparative study against an ImageNet-based encoder, also employing t-SNE for visualization. Furthermore, we deploy machine learning models—specifically, k-nearest neighbors (KNN) and support vector machines (SVM)—to evaluate these features in terms of accuracy, precision, and recall.

Our findings underscore the potent capabilities of CNNs and the advantages of transfer learning in handling a variety of image classification challenges. We observed that hyperparameter tuning plays a critical role in enhancing model performance. Moreover, the use of pre-trained networks was found to provide a significantly richer feature representation compared to models trained from scratch.

## Requirements to run the code
matplotlib         &nbsp;&nbsp;&nbsp;&nbsp; 3.7.1  <br/>
numpy              &nbsp;&nbsp;&nbsp;&nbsp; 1.22.4  <br/>
session_info       &nbsp;&nbsp;&nbsp;&nbsp; 1.0.0  <br/>
sklearn            &nbsp;&nbsp;&nbsp;&nbsp; 1.2.2  <br/>
torch              &nbsp;&nbsp;&nbsp;&nbsp; 1.13.1+cu116  <br/>
torchvision        &nbsp;&nbsp;&nbsp;&nbsp; 0.14.1+cu116  <br/>
tqdm               &nbsp;&nbsp;&nbsp;&nbsp; 4.65.0  <br/>
<br/>
<br/>
IPython            &nbsp;&nbsp;&nbsp;&nbsp; 7.9.0  <br/>
jupyter_client     &nbsp;&nbsp;&nbsp;&nbsp; 6.1.12  <br/>
jupyter_core       &nbsp;&nbsp;&nbsp;&nbsp; 5.3.0  <br/>
notebook           &nbsp;&nbsp;&nbsp;&nbsp; 6.5.3  <br/>

## Instruction on how to train/validate the model
```
> Clone the repo
    >> git clone https://github.com/varun30300/Comp6321_Project.git
> Upload the Ipython notebooks on google colab/Kaggle.
> Upload the datasets from the given google drive link and modify the datasets paths in the notebooks accordingly. For example:
    dataset 1 = '/kaggle/input/colorectal-canc/Dataset 1/Colorectal Cancer '
    dataset 2 ='/kaggle/input/prostate-cancer/Dataset 2/Prostate Cancer'
    dataset 3 = '/kaggle/input/animal-faces/Dataset 3/Animal Faces'
> The notebooks are now ready to run!
```
## Source Code Package In PyTorch
1) Task 1/hyperparameter-tuning.ipynb: This file contains the code for hyper parameter tuning of learning rate, batchsize etc on dataset 1 <br/>
2) Task 1/best-hyperparameters-train-and-test.ipynb: This file contains the code training and testing the best hyperparameters we got on dataset 1. Also the TSNE plots using the model from scratch on all datasets <br/>
3) Task 2/pre-trained-model.ipynb: This file contatins training and testing of dataset 1 using a pre trained model and TSNE plots on all three datasets using the pre trained model.


## Datasets
Colorectal Cancer:- https://onedrive.live.com/?authkey=%21ADmb8ZdEzwFMZoo&id=FB338EA7CF297329%21405133&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp
Prostate Cancer:-  https://onedrive.live.com/?authkey=%21APy4wecXgMnQ7Kw&id=FB338EA7CF297329%21405132&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp
Animal faces:-  https://onedrive.live.com/?authkey=%21AKqEWb1GDjWPbG0&id=FB338EA7CF297329%21405131&cid=FB338EA7CF297329&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp
---
