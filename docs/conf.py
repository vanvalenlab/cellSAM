from sphinx_gallery import scrapers
import napari

project = 'CellSAM'
copyright = '2025, vanvalenlab'
author = 'vanvalenlab'
release = '0.0.1-dev'

# NOTE: borrowed from https://github.com/napari/docs/blob/b9831f55e4c3aa012bc425a669a2ebee7abd87b7/docs/conf.py#L286C1-L311C67
def napari_scraper(block, block_vars, gallery_conf):
    """Basic napari window scraper.

    Looks for any QtMainWindow instances and takes a screenshot of them.

    `app.processEvents()` allows Qt events to propagateo and prevents hanging.
    """
    imgpath_iter = block_vars['image_path_iterator']

    if app := napari.qt.get_qapp():
        app.processEvents()
    else:
        return ""

    img_paths = []
    for win, img_path in zip(
        reversed(napari._qt.qt_main_window._QtMainWindow._instances),
        imgpath_iter,
    ):
        img_paths.append(img_path)
        win._window.screenshot(img_path, canvas_only=False)

    napari.Viewer.close_all()
    app.processEvents()

    return scrapers.figure_rst(img_paths, gallery_conf['src_dir'])


# NOTE: borrowed from https://github.com/napari/docs/blob/b9831f55e4c3aa012bc425a669a2ebee7abd87b7/docs/conf.py#L273
def reset_napari(gallery_conf, fname):
    from napari.settings import get_settings
    from qtpy.QtWidgets import QApplication

    settings = get_settings()
    settings.appearance.theme = 'dark'

    # Disabling `QApplication.exec_` means example scripts can call `exec_`
    # (scripts work when run normally) without blocking example execution by
    # sphinx-gallery. (from qtgallery)
    QApplication.exec_ = lambda _: None


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",  # Docstring format
    "sphinx.ext.autosummary",  # Reference guide
    "sphinx.ext.autodoc",  # Docstring summaries
    "sphinx_gallery.gen_gallery",  # Example gallery
]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "plot_gallery": "True",
    "image_scrapers": ("matplotlib", napari_scraper),
    "reset_modules": (reset_napari,),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
suppress_warnings=[
    # Warnings due to unpickleable objects in the sphinx-gallery conf (i.e the fns)
    # See sphinx-doc/sphinx#12300
    "config.cache",
]

# Generate autosummary pages
autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_title = "CellSAM"
html_static_path = ['_static']
