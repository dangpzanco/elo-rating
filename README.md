# A comprehensive analysis of the Elo rating algorithm

## Stochastic model, convergence characteristics, design guidelines, and experimental results

---

### Abstract

The Elo algorithm, due to its simplicity, is widely used for rating in sports competitions as well as in other applications where the rating/ranking is a useful tool for predicting future results. However, despite its widespread use, a detailed understanding of the convergence properties of the Elo algorithm is still lacking. Aiming to fill this gap, this paper presents a comprehensive (stochastic) analysis of the Elo algorithm, considering round-robin (one-on-one) competitions. Specifically, analytical expressions are derived characterizing the behavior/evolution of the skills and of important performance metrics. Then, taking into account the relationship between the behavior of the algorithm and the step-size value, which is a hyperparameter that can be controlled, some design guidelines as well as discussions about the performance of the algorithm are provided. To illustrate the applicability of the theoretical findings, experimental results are shown, corroborating the very good match between analytical predictions and those obtained from the algorithm using real-world data (from the Italian SuperLega, Volleyball League).

---

### How to run the code

- Download it! Anonymous Github doesn't support bulk download, so just run the following:

```bash
wget https://anonymous.4open.science/api/repo/elo-rating-CD35/file/data_utils.py
wget https://anonymous.4open.science/api/repo/elo-rating-CD35/file/environment.yml
wget https://anonymous.4open.science/api/repo/elo-rating-CD35/file/model_utils.py
wget https://anonymous.4open.science/api/repo/elo-rating-CD35/file/plot_discussion.py
wget https://anonymous.4open.science/api/repo/elo-rating-CD35/file/plot_experiments.py
wget https://anonymous.4open.science/api/repo/elo-rating-CD35/file/plot_utils.py
wget https://anonymous.4open.science/api/repo/elo-rating-CD35/file/run_code.sh
wget https://anonymous.4open.science/api/repo/elo-rating-CD35/file/utils.py
wget https://anonymous.4open.science/api/repo/elo-rating-CD35/file/utils_pytorch.py
wget https://anonymous.4open.science/api/repo/elo-rating-CD35/file/webscrape_csv.py
```

- Create a new Conda environment. In the case Conda is not installed, I suggest using the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution.

```bash
conda env create -n elo -f environment.yml
conda activate elo
```

- **IMPORTANT**: Setup the WebDriver for your default web browser with [this guide](https://www.selenium.dev/documentation/webdriver/getting_started/install_drivers/) from Selenium (a webcrapping library). This is required for downloading the dataset.

Then, either execute the script `run_code.sh`, or run the individual Python files included in the script:

#### Download dataset

This step opens up your web browser and automatically loads each season of the SuperLega on FlashScore's website. It should create a folder `dataset_csv`, where each season is saved as a separate CSV file.

```bash
python webscrape_csv.py
```

#### Convert dataset to Zarr groups

Before running the experiments, the dataset is first converted to Zarr arrays and saved on the `superlega` folder. This makes it easy to handle the data, since Zarr exposes a Numpy-like interface (see the [Zarr docs](https://zarr.readthedocs.io/en/stable/) for more info).

```bash
python data_utils.py
```

#### Plot the figures

All the figures included in the paper are generated and saved on the `figures` folder. Note that the `plot_discussion.py` script doesn't require the dataset to be downloaded.

```bash
python plot_discussion.py
python plot_experiments.py
```

Everything was tested on Linux, but should work fine on Windows/Mac as well.
