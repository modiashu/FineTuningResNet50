File Name: Readme
Description: This file describes steps on how to use trained model to predicts objcet states

Package contains:
	1. Train.py : Training script used to learn neural network
	2. Test.py : Testing script to check performance on testing data
	3. Readme.txt
	
	Model and weights files are lcated on GAIVI at path
	:/home/modia/Spring2018/1project/Project1
	
	4. ResNetModel.h5 : Model learned for custom data
	5. Weights.hdf5 : Weights file learned on custom data
				  I have attached two formats of weight files 1) weights_1.h5 2) weights_2.hdf5
				  Please use weights_2.hdf5 while testing the performance. If you are unable to 
				  load this file then use weights_1.h5
	
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
- Use weight file as "Weights.hdf5"
- Model saved is named as "ResNetModel.h5"

Note:
- I have attached two formats of weight files 1) weights_1.h5 2) weights_2.hdf5
  Please use weights_2.hdf5 while testing the performance. If you are unable to 
  load this file then use weights_1.h5
- Please modify Test.py accordingly
