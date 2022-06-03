# Infinite Recommendation Networks ($\infty$-AE)
<!-- $\infty$-AE's implementation in JAX. Kernel-only method outperforms complicated SoTA models with a closed-form solution and a single hyper-parameter. -->

This repository contains the implementation of $\infty$-AE from the paper "Infinite Recommendation Networks: A Data-Centric Approach" [[arXiv]]() where we leverage the NTK of an infinitely-wide autoencoder for implicit-feedback recommendation. Notably, $\infty$-AE:
- Is easy to implement (<50 lines of relevant code)
- Has a closed-form solution
- Has only a single hyper-parameter, $\lambda$
- Even though simplistic, outperforms *all* complicated SoTA models

If you find any module of this repository helpful for your own research, please consider citing the below under-review paper. Thanks!
```
@article{sachdeva2022b,
  title={Infinite Recommendation Networks: A Data-Centric Approach},
  author={Sachdeva, Noveen and Dhaliwal, Mehak Preet and Wu, Carole-Jean and McAuley, Julian},
  journal={arXiv preprint arXiv:2206.XXXXX},
  year={2022}
}
```

**Code Author**: Noveen Sachdeva (nosachde@ucsd.edu)

---

## Setup
#### Environment Setup
```bash
pip install -r requirements.txt
```

#### Data Setup
Once you've correctly setup the python environment, the following script will download the ML-1M dataset and preprocess it for usage:

```bash
./setup.sh
```

---
## How to train $\infty$-AE?
- Edit the `hyper_params.py` file which lists all config parameters of $\infty$-AE. Note that $\lambda$ is currently grid-searched in `main.py` so changing it will bring no difference, until the code is adjusted.
- Finally, type the following command to train and evaluate $\infty$-AE:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

---
## Results sneak-peak

Below are the nDCG@10 results for the datasets used in the [paper]():

| Dataset           | PopRec  | MF    | NeuMF  | MVAE  | LightGCN    | EASE  | $\infty$-AE | 
| -------           | ------  | --    | -----  | ----  | --------    | ----  | ----------- |
| Amazon Magazine   | 8.42    | 13.1  | 13.6   | 12.18 | 22.57       | 22.84 | **23.06**   |
| MovieLens-1M      | 13.84   | 25.65 | 24.44  | 22.14 | 28.85       | 29.88 | **32.82**   |
| Douban            | 11.63   | 13.21 | 13.33  | 16.17 | 16.68       | 19.48 | **24.94**   |
| Netflix           | 12.34   | 12.04 | 11.48  | 20.85 | *Timed out* | 26.83 | **30.59***   |

*Note*: $\infty$-AE's results on the Netflix dataset (marked with a *) are obtained by training only on 5% of the total users. Note however, all other methods are trained on the *full* dataset.

---

## MIT License