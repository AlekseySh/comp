# Instruction to reproduce 3d place solution (team: gusi-lebedi).


## Part 1. Feature engeneering
In our sulution we used base data (`train.csv`, `road_segments` and `Sanral_v3`)
and also we used additional data
shared between all participants (`Weather Data`, `Public holidays` and `Uber Movement Data`).
Note, that we didnt't use 
data leak which was found in `Injuries*.csv` and `Vehicles*.csv`.

We did a lot of manual markup and store this files in `data` folder,
we marked this files and directories with `_processed` suffix in their names.


### SANRAL
Folder: `data/SANRAL_processed`.

We automaticly assigned each `CCTV`, `VDS` and `VMS` to the nearest road segment and than for each
road segment counter number of assigned cameras - `cameras_count.csv`. 

Also we convert original `SANRAL` data to more usefull format:
* `cameras_commissioning_dates.csv` based on 
`Commissioning Date` column from `VDS_locations.xlsx`
* `vds_locations.csv` is the second sheet of `SANRAL_V3/VDS_locations.xlsx` with handled missing records.
* `VDS_hourly_all.csv` is concatenation of all hourly files from `SANRAL_V3/Vehicle detection sensor (VDS)/[2016-2019]`.

### Road segments
Folder: `data/road_segments_processed`.

We opened shapefiles and the attributes table in `QGis` (free programm for viewing geo files) and 
created 3 more columns:
1. `main_route` - we matched each road segment with long part of road named `route#` (see image below).
2. `vds_id` - the neareast VDS camera id to the road segment.
3. `num` - the enumarate id of the road segment.

![](data/routes.png)


### Weather
Folder: `data/weather_processed`.

We downloaded 2 airport weather files 
(`68816.01.01.2016.31.03.2019.1.0.0.en.utf8.00000000.csv` and 
`FACT.01.01.2016.31.03.2019.1.0.0.en.utf8.00000000.csv`)
 from provided site and combined them into 1 table with small preprocessing
(see `weather.csv` folder).


### Clean train.csv
We found that some accidents have only day indicator (with missing hour), so we decided to inpute this 
missed values with ones taken from `Injures*.csv` / `Vehicles.csv` files. As a result we replaced roughly
1500 records and stored fixed table as `train_cleaned.csv`.
Please note, as in the `train.csv` there is only 2017-2018 year data, we didn't use the leak
from 2019 year.


### Uber data
Folder: `data/uber_processed/`

We downloaded a lot of files from
`Uber Movement` site provided by orginizers (see `uber_files_downloaded.zip`).
Than we filtered and 
combined them into 1 large table - `uber_files_joined.csv`.

Since the travel time data is related to uber zones, we
mannualy created mapping between uber zones and road segments - `enum_to_uzone.json`.
In addition, we had to manually determine which road segments a car consistently
 crosses moving from one uber zone to another (`routes.json` and `sid_neighbors.json`).
 
For convenience, we store lengths of segments in `routes_length.json` and 
replaced all road segment names with numbers (`sid_to_enum.json`) using
`num` column from shapefile from the previous step. As a result we got the travel times in
submit format: for each road segment for each timestmap - `segment_ttime_daily.csv`.


### Downaload resulted TRAIN and TEST files
To avoid wasting time on generating and formatting files,
you can download ones: [todo1](link1.com) and [todo2](link2.com).
You can easily verify that the data in the tables doesn't contain data leak and
data from datasets that aren't available to other participants. Please, after downloading 
this files just store
them as `data/train*.pkl` and `data/test*.pkl`.
 

## 2. Model training

Training model is much simpler than feature engeneering: we trained single neural network 
(we used `fastai` framework for this purpose) without
any ensembling using 1-fold local validation.

* To train the model you have to run `notebooks/modeling_fai.ipynb`, it requires `train.pkl` 
and `test.pkl` as inputs,
generated (or downloaded) on the previous step. As an output, this will give the 
submit file and the model weights.
Estimated time of model training on nvidia-tesla k80 is about 30 min.
