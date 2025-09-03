__version__ = "0.0.dev1"

from .cellsam_pipeline import cellsam_pipeline
from .model import segment_cellular_image, get_model, get_local_model
from .sam_inference import CellSAM


def download_training_data(version=None):
    """Download the data for the CellSAM model.

    The compressed dataset will be downloaded to ``$HOME/.deepcell/data``.

    Parameters
    ----------
    version : str, optional, default=latest
       Which version of data to download. If not specified, downloads the latest
       published dataset.
       Currently available versions:

        - 1.2 (latest)
        - 1.0
    """
    from . import _auth

    version = "1.2" if version is None else version
    record = _auth._data_versions[version]

    _auth.fetch_data(
        record["asset_key"], cache_subdir="datasets", file_hash=record["asset_hash"]
    )
