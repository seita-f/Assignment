# Environment
- MacOS Montery (Version 12.7.6)
- VSCode
- github for version control (private repo)

# Setup
1. Build an image, create a container
Run under the assignment directory
```
$ cd Assignment
$ docker compose up -d  
```

2. Run a container
```
$ docker run -it --rm assignment:v1
```
or <br> 
Reccomed:
```
$ docker run -it --rm -v <PATH>/Assignment:/Assignment assignment:v1
# Ex: docker run -it --rm -v /Users/fujiwaraseita/Desktop/Assignment:/Assignment assignment:v1
```

# Feature Extraction
Feature extraction with file saved you defined in config
```
python -m src.features.main
```
Feature extraction with file saved you provide
```
python -m src.features.main --save-file datasets/covid19_feature_extraction/test.csv
```

In addition to that, you can select features you want to add by commentig out the item from FEATURE_REGISTRY in config
Ex: Skip adding population and health expenditure features
```
features_to_apply:
  - TimeDelayFeatures
  - DayFeatures
  - DistanceToOriginFeatures
  - CountryAreaFeatures
  # - CountryPopulationFeatures
  - CountrySmokingRateFeatures
  - CountryHospitalBedsFeatures
  # - CountryHealthExpenditureFeatures
```

# Training
It does loading data, feature extraction, and training model at once.
The file for feature extraction will be saved with the default name you define in config file
```
python -m src.models.train_model
```
It loads the extraction file you provide and train the model
```
python -m src.models.train_model --features datasets/covid19_feature_extraction/test.csv
```

# Inference (Not completed*)
```
python -m src.models.inference --test datasets/covid19_global_forecasting_week_1/test.csv
```

# Note
Firstly, I would like to finish implementing inference part, then visualize the forecast results with streamlit and also use MLFlow to easily compare model parameters, accuracy, and other metrics in the future as well as implementing unit test and github actions. It was a bit challenging but I enjoyed this assignement!


