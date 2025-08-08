Model and Datasets
==================

Pre-trained models and training datasets are licensed under a 
[modified Apache license][license] for non-commercial academic use only.
An API key for accessing datasets and models can be obtained at <https://users.deepcell.org/login/>.

[license]: https://github.com/vanvalenlab/cellSAM/blob/master/LICENSE.md

API Key Usage
-------------

The token that is issued by <https://users.deepcell.org> should be added as an
environment variable:

```bash
export DEEPCELL_ACCESS_TOKEN=<token-from-users.deepcell.org>
```

This line can be added to your shell configuration (e.g. ``.bashrc``, ``.zshrc``,
``.bash_profile``, etc.) to automatically grant access to DeepCell models/data
upon login.

(download_models)=
Model Weights
-------------

Pre-trained model weights are required to run the inference pipeline. These will
be automatically downloaded to ``$HOME/.deepcell/models`` the first time you
run :func:`cellsam_pipeline`.

Alternatively, you can call ``get_model`` without any arguments to download the
latest pre-trained model weights.

Training Data
-------------

```{warning}
The training dataset is around 14GB - make sure you have space and sufficient
network bandwidth before attempting to download.
```

Similarly, training data can be downloaded for local use with:

```python
>>> from cellSAM import download_training_data

>>> download_training_data()
```
