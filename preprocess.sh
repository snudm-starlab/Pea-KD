# Code for preprocess glue data
python ./src/data/download_glue_data.py --data_dir data_raw --tasks all \

mv data_raw ./src/data/
