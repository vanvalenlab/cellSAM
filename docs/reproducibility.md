# Reproducibility

In addition to the `cellSAM` library, the source repo contains additional
code to aid in reproducing the results in the publication.
Additional code for reproducibility can be found in [`paper_evaluation`][gh-eval]

[gh-eval]: https://github.com/vanvalenlab/cellSAM/tree/master/paper_evaluation

Additional resources including pre-trained model weights and the evaluation dataset
are required for reproducibility.
All necessary components are available for download - see {ref}`download_models` for
details.

## Setup

In a new (empty) virtual environment, install cellSAM from the parent directory.

````{tip} Example: creating an environment
Users are encouraged to use whichever environment management system with which they
are most comfortable (`uv`, `pixi`, `conda/mamba`, etc.)

For those unsure, Python's [built-in environment management module][python-venv] is
a simple, ubiquitous option. For example, to create and enter a new environment:

```bash
python3.XX -m venv cs-eval-env
source cs-eval-env/bin/activate
```

Where `XX` is the Python version you wish to use (e.g. `python3.13`).
You can then verify the newly created environment is empty (though `pip` should be available):

```bash
pip list
```
````

[python-venv]: https://docs.python.org/3/library/venv.html

For example, from the `paper_evaluation` directory:

```bash
pip install ..
```

### Evaluation dependencies

Once in a "clean" environment, install the requirements for the evaluation suite:

```bash
pip install -r requirements.txt
```

```{note}
This may downgrade some of the dependencies (e.g. `torch`, `numpy`, etc.) installed
in the previous step.
```

### Evaluation models

The pretrained model weights necessary for reproducibility are available via the `get_model`
function:

```python
>>> from cellSAM import get_model
>>> get_model();
```

This will automatically download and unpack the latest version of the pretrained model weights.

```{note} Model versions
You may use the `version=` keyword argument for `get_model` to specify a specific
model version for evaluation.

 - Version `1.2` is the version that was used to produce the published results in the paper
 - Version `1.2` is the *minimum* model version which is designed to work with the
   reproducibility workflow.
```

### Evaluation dataset

Make sure you have the evaluation dataset. This can be downloaded with:

```python
>>> from cellSAM import download_training_data
>>> download_training_data()
```

This will initiate the download of a compressed data archive.
The compressed data will be downloaded to
`$HOME/.deepcell/datasets/cellsam-data_v{X.Y}.tar.gz` where `X.Y` is the requested
dataset version.
See {ref}`download_models` for details.
Once the download is complete, unpack/inflate the dataset to a desired location.

````{caution} Dataset Size
The compressed data archive is 14GB in size, and inflates to 84GB when uncompressed.
Therefore you may want to unpack the data to different location.
Similarly, the decompression is comuptationally intensive, and may benefit from parallel
decompression algorithms.
Here's an example incantation which will store the unpacked dataset to `/data` using 8
threads for decompression:

```bash
tar --use-compress-program="unpigz -p 8" $HOME/.deepcell/datasets/cellsam-data_v1.2.tar.gz -C /data
```

The unpacked data will then be available at `/data/cellsam_v1.2`.
````

### Running the evaluation

Once all of the above steps are complete, the evaluation can be run via the `all_runsh.sh`
shell script:

```bash
./all_runs.sh
```

The results of each run will be saved locally in a `summary.csv` that records the datset,
model used, and `f1_mean` for that run.

#### Individual evaluations

It is not necessary to run the entire evaluation suite - evaluation can be limited to
specific datasets.
See the `all_runs.sh` for a general idea of how to do so via `eval_main.py`.
