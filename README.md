# trashnet

# <a id="2"></a> Introduction
## <a id="2.1"></a> Data Introduce
* Download the dataset https://drive.google.com/drive/folders/0B3P9oO5A3RvSUW9qTG11Ul83TEE
* Pre-trained model https://drive.google.com/file/d/1j-BRmIJeNSa80yFAn5JxA_Ddv3FTwqWP/view?usp=sharing

## <a id="2.2"></a> Requirement
* Tensorflow-gpu 2.1  scikit-learn  opencv-python  matplotlib

## <a id="2.3"></a> Preprocess
run python train_val_test.py   to split dataset

# <a id="3"></a> Train 
run python train.py --model_name densenet121 --batch_size 128 --input_size 128 --epoch 50

# <a id="3"></a> Predict 
run python train.py --model_name densenet121 --input_size 128

# <a id="4"></a> Confusion Matrix

