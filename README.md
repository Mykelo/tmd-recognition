# How to run

1. Create conda environment:

```sh
conda create -n tmd-recognition python=3.10 poetry
```

2. Activate the environment:

```sh
conda activate tmd-recognition
```

3. Install dependencies:

```sh
poetry install
```

# Structure

The source code is divided into 4 parts: 

- `src/images` - containing code responsible for processing and analyzing images
- `src/videos` - containing code responsible for processing and analyzing videos
- `src/surveys` - containing code responsible for processing and analyzing surveys
- `src/common` - containing all common components

## Images

There are 2 main entry points to this part of the code:

- `src/images/experiments.ipynb` - notebook with experiments
- `src/images/analysis.ipynb` - notebook that generates charts


## Surveys

Here we have 3 notebooks responsible for:

- `src/surveys/Create csv.ipynb` - processing raw data into data suitable for a decision tree,
- `src/surveys/Tree.ipynb` - implementation and testing of the decision tree algorithm,
- `src/surveys/rm_bad_patiences.ipynb` - filter out incorrect data.