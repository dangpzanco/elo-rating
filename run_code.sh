#!/bin/bash

### RUN THIS OUTSIDE THE BASH SCRIPT:
# conda env create -n elo -f environment.yml; conda activate elo

################################################

# Download dataset
python webscrape_csv.py

# Convert dataset from CSV to Zarr for easier handling
python data_utils.py

# Generate all the figures (both in English and Portuguese)
python plot_discussion.py
python plot_experiments.py

