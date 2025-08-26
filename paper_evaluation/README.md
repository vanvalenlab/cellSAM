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
