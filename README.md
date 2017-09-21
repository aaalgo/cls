# A developed deep learning architecture for classification

## Requirements
- Python
- Tensorflow
- Picpac

## Importing database
### picpac-import
#Stream data into same image database format  
Eg:  
picpac-import -f 2 ImageDirectory db  
#ImageDirectory contains N subdirectories named 0, 1, ..., each containg images for one category  
#Refer to [picpac] (#https://github.com/aaalgo/picpac/blob/master/README.md) for more info  

## Training
### cls-train.py
#Trainer of images classification, allows evaluation during training
```
arguments:  
--db  
		Training image database  
--classes  
		Numbers of categories of classification  
--channels  
		Numbers of channels used in training  
optional arguments:  
--opt  
        Optimizer of network training, choices of adam(default) or gradient  
--test_db  
        Evaluating image database  
--model  
        Directory to save models  
--learning_rate  
		Initial learning rate  
--test_steps  
		Number of steps to run evaluation  
--save_steps   
		Number of steps to save model  
--max_steps  
		Number of steps to run trainer  
--split  
		Split train image into this number for cross-validation  
--split_fold  
		Part index for cross-validation  
Eg:   
./cls-train.py --db db.train --test_db db.test --classes 2 --channels 1  
./cls-train.py --db db.train --split 5 --split_fold 3  
```
### fcn-cls-train.py
#Trainer of segmented images classification  
```
arguments:  
--pos  
		Training image dataset with positive part labelled  
--neg  
		Training image dataset without labelling  
Other arguments have the same usage as cls-train.py  
```
## Evaluating
### cls-predict.py
#Evaluation of classification using cls-train.py  
```
arguments:  
--input  
		Input directory of images, can be any directory  
--model  
		Model saved during training  
--channels  
		Channels of images used  
Eg:  
./cls-predict.py --input ImageDirectory --model model/200000 --channels 1  
```
### fcn-cls-val.py
#Evaluation of classification using fcn-cls-train.py  
```
arguments:  
--db  
		Evaluation image database  
```

