# tbd-project-25

Placeholder repository for Team 25 – update with final project title and description when available

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Abstract

A concise summary of the project's goals, the problem it addresses, and its intended audience. This section can include potential use cases and key features.

## Download datasets
Download data from [https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009337#sec028](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009337#sec028) S1 File.

## Installation

Python environment
```
python -m venv metro
pip install torch pandas tqdm pyyaml
pip install -I -e .
...
```

### R environment
```bash
mamba install -c conda-forge r-base r-rcpp r-igraph r-biocmanager
```

```R
BiocManager::install("mixOmics")
```

## Quick Start

Given a metabolic or transcriptomic profile, MeTrO can encode either to a mutually shared latent space and then decode it into the other (or back to the original in a probabilistic manner).

The following workflow is planned:

```python
import my_project

model = my_project.load_model(state_dict.pkl)
z = model.encode(metabol_profile)
recon_metabol = model.decode_m(z)
recon_transcript = model.decode_t(z)
```

```r
# Example usage (R)
library(my_project)

demo <- example_function()
print(demo)
```

## Usage

This tool is run as a set of scripts from the command line. The primary control method of training the VAE in this package is a set of config files in `config/`. See `config/default.yml` for the comprehensive set of model and training parameters and what they may look like. Any specified subset of these parameters can be overwritten by providing a custom config. For example:

```
python scripts/run.py -c config/test.yml
```
The above code will train a VAE for a single epoch on a random 1% subset of the data used to train the full model. For the standard settings of a full VAE training run:
```
python scripts/run.py -c config/control.yml
```
```r
# More usage examples (R)
library(demoProject)

demo <- advanced_function(parameter1 = "value1")
print(demo)
```

## Contribute

Contributions are welcome! If you'd like to contribute, please open an issue or submit a pull request. See the [contribution guidelines](CONTRIBUTING.md) for more information.

## Support

If you have any issues or need help, please open an [issue](https://github.com/hackbio-ca/demo-project/issues) or contact the project maintainers.

## License

This project is licensed under the [MIT License](LICENSE).
