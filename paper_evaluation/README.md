# Setup

In a new (empty) virtual environment, install cellSAM from the parent directory.
For example, if you are in the `paper_evaluation` directory:

```
pip install ..
```

Then install the requirements for the evaluation suite - NOTE: this may downgrade
some of the dependencies (e.g. `torch`, `numpy`, etc.) installed in the previous
step.

```
pip install -r requirements.txt
```

Make sure you have the evaluation dataset. This can be downloaded with:

```python
>>> from cellSAM import download_training_data
>>> download_training_data()
```

This will initiate the download of the tarred-gzipped datasets for evaluation.
Once the download is complete, unpack/inflate the dataset:

```
cd .deepcell/data
tar -xzf cellsam-dataset_v1.0.tar.gz
```

NOTE: This inflates the data from ~14GB to ~84GB and may take several minutes.
If installed, `pigz` can be used to significantly reduce decompression times.

The final result is the cellsam evaluation dataset found at the path
`$HOME/.deepcell/data/dataset`.

TODO: Evaluation models
 - Need the named models to be found at `$HOME/.deepcell/models`
 - Proposal: package the 3 models up into an archive called `cellsam-evaluation`
   that can be accessed from users.deepcell.org and lives at
   `$HOME/.deepcell/models/cellsam-evaluation` locally.
