File Name: Readme
Description: This file describes steps on how to use trained model to predicts objcet states

Package contains:
	1. Train.py : Training script used to learn neural network
	2. Test.py : Testing script to check performance on testing data
	3. Readme.txt
	
Steps:
- Create folder named 'test'
- Create sub folders in 'test' folder as below
	- creamy_paste
	- diced
	- grated
	- juiced
	- jullienne
	- sliced
	- whole
- Place corresponding images in these folders.
- Run Test.py file get prediction accuracy.
- Use weight file as "Weights.hdf5" (Have not uploaded this weight file here on gitHub)
- Model saved is named as "ResNetModel.h5" (Have not uploaded this weight file here on gitHub)

Note:
- Please modify Test.py accordingly to load pre trained weights
