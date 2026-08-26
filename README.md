# fitsedit

A command-line tool for masking FITS images.

## Install

Clone the repo, then (recommended: inside a virtual environment) install it:

```bash
git clone https://github.com/jnpippert/fitsedit.git
cd fitsedit
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This puts the `fitsedit` command on your PATH.

## Usage

```bash
fitsedit image1 image2 ...
fitsedit -m s -z 2 # starts in satellite mode and 2x zoom
```
