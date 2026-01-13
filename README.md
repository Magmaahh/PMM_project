# PMM Course Project
Project for the Process Mining and Management course (a.y. 2025–2026) offered by the University of Trento.

This repository contains the implementation of **Project Typology #2: “Extracting and evaluating recommendations”**.

The goal of the project is to use predictive process monitoring techniques to both predict process outcomes and derive recommendations that can guide ongoing process executions towards a desirable outcome.

## Table of contents
- [Requirements](#requirements)
- [Project structure](#project-structure)
- [How to run](#how-to-run)
- [Authors](#authors)

## Requirements

Create and activate a virtual environment:

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (Powershell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then install the required Python packages:
```bash
pip install -r requirements.txt
```

## Project structure
All main files are contained in the `src/` folder:
- ```main.py```: entry point and pipeline controller (runs the full workflow).
- ```config.py```: configurable constants used across the modules.
- ```utils.py```: data loading and preprocessing utilities.
- ```functions.py```: core logic (training/testing, recommendation extraction, and evaluation).

In addition, the repository includes:
- ```data/```: to contain both raw data and processed data used for training and testing.
- ```fig/```: to store plots of the trained decision trees.

### Constants definition
Constants used in the project can be modified in ```src/config.py```.

Paths to store training and testing datasets are defined by:
```bash
TRAIN_DATA_PATH
TEST_DATA_PATH
```

While processed logs are stored in the folder defined by ```PROCESSED_DATA_PATH```.

In addition to these, it is also possible to configure:

- ```TREES_PLOTS_PATH```, the folder that will contain the plots for the trained decision trees

- The set of ```PREFIX_LENGTH``` values, to compare results across different prefix sizes.

- The model hyperparameters search space (via the ```params``` dictionary), to extend the Grid Search optimization process.

### utils.py - data preparation module
This module processes given training and testing logs, and produces:
- Boolean-encoded prefix traces used for training and testing.
- Boolean-encoded full-length test traces used for recommendations evaluation.

More specifically, it performs:
- Raw data extraction (full length traces and activity list).
- Boolean encoding of extracted traces.

### functions.py - execution module
This module contains the main functions used to execute the project, including:
- Decision tree hyperparameters optimization.
- Decision tree training and testing.
- Decision tree visualization and saving (both as code and as a figure).
- Positive-paths extraction from the decision tree.
- Recommendations extraction for each test prefix.
- Recommendations evaluation against the corresponding full-length traces.

> [!NOTE]
> The process is executed for each ```PREFIX_LENGTH``` value set in ```src/config.py```.

## How to run
To execute the full pipeline, run:
```bash
python3 src/main.py
```

> [!NOTE]
> **Be sure to run the command from the project root directory** (i.e., the folder containing this README) to ensure all relative paths resolve correctly.

During execution, logs describing each major step of the pipeline are printed to the terminal.

At the end, the script prints, for each ```PREFIX_LENGTH``` value:

- Prediction performance on test prefixes.

- Recommendation evaluation results.


## Authors
| Name                | Email                                 |
|---------------------|---------------------------------------|
| Lorenzo Fasol     | lorenzo.fasol@studenti.unitn.it      |
| Stefano Camposilvan | stefano.camposilvan@studenti.unitn.it |
