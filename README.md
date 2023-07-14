# Analysis of the Elo rating algorithm in round-robin tournaments

## Stochastic model, design guidelines, and experimental results

---

### Abstract

The Elo algorithm, renowned for its simplicity, is widely used for rating in sports tournaments and other applications. However, despite its widespread use, a detailed understanding of the convergence characteristics of the Elo algorithm is still lacking. Aiming to fill this gap, this paper presents a comprehensive (stochastic) analysis of the Elo algorithm, considering round-robin (all meet all) tournaments. Specifically, analytical expressions are derived describing the evolution of the skills and performance metrics. Then, taking into account the relationship between the behavior of the algorithm and the step-size value, which is a hyperparameter that can be controlled, some design guidelines and discussions about the performance of the algorithm are provided. Experimental results are shown confirming the accuracy of the analysis and illustrating the applicability of the theoretical findings using real-world data (from the Italian SuperLega, Volleyball League).

---

### How to run the code

- Clone/download the repository

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

#### Convert dataset to Zarr groups/folders

Before running the experiments, the dataset is first converted to Zarr arrays and saved on the `superlega` folder. This makes it easier to handle the data, since Zarr exposes a Numpy-like interface (see the [Zarr docs](https://zarr.readthedocs.io/en/stable/) for more info).

```bash
python data_utils.py
```

#### Plot the figures

All the figures included in the paper are generated and saved in the `figures` folder. Note that the `plot_discussion.py` script doesn't require the dataset to be downloaded.

```bash
python plot_discussion.py
python plot_experiments.py
```

Everything was tested on Linux, but should work fine on Windows/Mac as well.
