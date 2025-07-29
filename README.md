# Thesis repository for Jonny Aarstad Igeh's Master of Science: Computational Physics

## Document files
All document files are located in the /doc folder, and the general latex document structure is found in main.tex. All figures are located in doc/figs, all chapters in doc/chapters etc. The finalized thesis is at doc/main.pdf. 


## Program files
The program is built upon the potential functions and quantum dot functions located in src/utils, potential.py and qd_system.py respectively. These two scripts are at the base of all subsequent scripts. 

The two major scripts that perform the optimization and time-evolution of the Morse double-well system introduced in the thesis are found at src/sinc_optimization.py and src/sinc_time_evolution.py, with running examples in the if_name blocks. It should be intuitive to testrun the scripts after reading chapter 3 in the thesis. Intermediate results throughout the thesis have been achieved with the supplementary scripts also found in the same folder. We believe the file-names should clearly indicate what they do for the educated reader (after reading the thesis).

The src/data folder contains _some_ of the data used throughout, but due to GitHub storage restriction not all data is available. It can be easily produced by the user after installing the code framework.


### Prerequisites

- Python 3.x
- [Poetry](https://python-poetry.org/docs/) for dependency management and packaging.

### Installation

To build the project and install dependencies, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/Thesis.git
cd Thesis
```

2. Install the project dependencies with Poetry:

```bash
poetry install
```

This will create a virtual environment and install all the necessary packages as specified in `pyproject.toml`.

### Using the `src` Package

After installing the dependencies, you can install the `src` package within the virtual environment created by Poetry. This allows you to import the utility functions in other projects or files.

To install the `src` package in editable mode, run:

```bash
poetry run python -m pip install -e .
```