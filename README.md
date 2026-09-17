# maskfits

A PySide6 GUI for masking pixels in FITS astronomical images.

## Install

```bash
pip install maskfits
```

This puts the `maskfits` command on your PATH.

### From source

To work on maskfits itself, clone the repo and install it editable
(recommended: inside a virtual environment) instead:

```bash
git clone https://github.com/jnpippert/maskfits.git
cd maskfits
python3 -m venv .venv # optional
source .venv/bin/activate # optional
pip install -e .
```

## Usage

```bash
maskfits image1 image2 ...
maskfits -m s -z 2 # starts in satellite mode and 2x zoom
```
