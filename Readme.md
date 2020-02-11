# Instruction to reproduce 3-d place solution (team: gusi-lebedi).


### 1. Feature engeneering
In our sulution we used base dataset (`train.csv`, `road_segments` and `Sanral`) and also we used additional data
shared between all participants (`Weather Data`, `Public holidays` and `Uber Movement Data`). Note, that we didnt't use 
data leak: we removed all 2019 related records from `Injuries.csv` and `Vehicles.csv`.

* TODO
* TODO


* Hint: to avoid wasting time on generating the files, you can download ones: [todo1](link1.com) and [todo2](link2.com).
You can easily verify that the data in the tables doesn't contain data leak and data from datasets that aren't available
 to other participants.
 

### 2. Model training

Training model is much simpler than feature engeneering: we trained single neural network 
(we used `fastai` framework for this purpose) without
any ensembling using 1-fold local validation.

* To train the model you have to run `notebooks/modeling_fai.ipynb`, it requires `train.pkl` and `test.pkl` as inputs,
generated (or downloaded) on the previous step. As an output, this will give the submit file and the model weights.
Estimated time of model training on nvidia-tesla k80 is about 30 min.
