__version__ = "0.0.dev1"

from .cellsam_pipeline import cellsam_pipeline
from .model import segment_cellular_image, get_model, get_local_model
from .sam_inference import CellSAM


def download_training_data():
    """Download the training data for the CellSAM model.

    The compressed dataset will be downloaded to ``$HOME/.deepcell/data``.
    """
    from ._auth import fetch_data

    asset_key = f"data/cellsam/cellsam-dataset_v1.0.tar.gz"
    asset_hash = "848e9da232a82893f07c95f60b54de02"
    fetch_data(asset_key, cache_subdir="data", file_hash=asset_hash)
