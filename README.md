# Defect classificaiton using NEU dataset

## Dataset
[NEU surface defect database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)

Samples with surface defects are merged into a single category (folder) call __MT_Defects__. 

Cleaned-up dataset is stored in folder __data__

## Data preparation
Call: python data_prep.py after modifying corresponding folders and split percentages for train/val/test. 

## Train/test 
Call: python defect_classify.py 

For parameters see inside the code. 