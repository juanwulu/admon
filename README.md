# Adaptive Deep Modularity Network (ADMoN)

[![license](https://img.shields.io/badge/license-BSD_3--Clause-gold.svg)](https://github.com/ChocolateDave/a2sos/blob/master/LICENSE)

ADMON is short for Adaptive Deep Modularity Network, a data-driven graph
model which is capable of simultaneously learning airport connections and clustering based on the delay patterns.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies locally.

```bash
git clone -b master --depth 1 https://github.com/ChocolateDave/admon.git
cd admon & pip install -e .
```

## Usage

Use the `fit_model.py` script file to train your own model. The data we use for our final report is proprietary and hence removed from the open-source repository. Please use `-h` flag to see all the valid hyperparameters for the model.

```bash
python admon/fit_model.py ARGS
```

## Citation

If you find this source code helpful, it'll be really helpful if you can cite our work.

```bibtex
@techreport{Juanwu_Ying_Admon2021,
  author      = {Juanwu Lu, Ying Zhou},
  title       = {Identify Airport Connections Based on Delay Patterns: A Data-driven Graph Clustering Approach},
  institution = {University of California, Berkeley},
  year        = {2021},
  type        = {techreport},
  month       = {5},
  url         = {https://github.com/ChocolateDave/admon},
  urldate     = {2021-05-01}
}
```

## License

This project is licensed under the [BSD 3-Clause License](./LICENSE)
