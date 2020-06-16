# Trashnet

# <a id="1"></a> Data Introduce
* Pre-trained model https://drive.google.com/file/d/1j-BRmIJeNSa80yFAn5JxA_Ddv3FTwqWP/view?usp=sharing
 unzip to models
* Download the dataset https://drive.google.com/drive/folders/0B3P9oO5A3RvSUW9qTG11Ul83TEE
unzip the dataset_resized.zip and put in the root dir

# <a id="2"></a> Requirement
* Tensorflow-gpu 2.1  
* scikit-learn  
* opencv-python  
* matplotlib

# <a id="3"></a> Preprocess
Split the dataset to test, val and test 
```
python train_val_test.py
```
# <a id="4"></a> Train
Train DenseNet121 model, set batch size to 128, input size 128 and 50 epoch
```
python train.py --model_name densenet121 --batch_size 128 --input_size 128 --epoch 50
```
# <a id="5"></a> Predict 
Test the trained model by images in test folder
```
python test.py --model_name densenet121 --input_size 128
```
# <a id="6"></a> Confusion Matrix
Confusion matrix image is saved in the root dir
# <a id="7"></a> Result
Result is saved in the root dir

