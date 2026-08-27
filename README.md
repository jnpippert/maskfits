# maskfits

A command-line tool for masking FITS images.

## Install

Clone the repo, then (recommended: inside a virtual environment) install it:

```bash
git clone https://github.com/jnpippert/maskfits.git
cd maskfits
python3 -m venv .venv # optional
source .venv/bin/activate # optional 
pip install -e .
```

This puts the `maskfits` command on your PATH.

## Usage

```bash
maskfits image1 image2 ...
maskfits -m s -z 2 # starts in satellite mode and 2x zoom
```
