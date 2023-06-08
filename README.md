# Infinite Recommendation Networks (∞-AE)

This repository contains the implementation of ∞-AE from the paper "Infinite Recommendation Networks: A Data-Centric Approach" [[arXiv]](https://arxiv.org/abs/2206.02626) where we leverage the NTK of an infinitely-wide autoencoder for implicit-feedback recommendation. Notably, ∞-AE:

- Is easy to implement (<50 lines of relevant code)
- Has a closed-form solution
- Has only a single hyper-parameter, $\lambda$
- Even though simplistic, outperforms *all* complicated SoTA models

The paper also proposes Distill-CF: how to use ∞-AE for data distillation to create terse, high-fidelity, and synthetic data summaries for model training. We provide Distill-CF's code in a separate [GitHub repository](https://github.com/noveens/distill_cf).

If you find any module of this repository helpful for your own research, please consider citing the below paper. Thanks!

```
@article{inf_ae_distill_cf,
  title={Infinite Recommendation Networks: A Data-Centric Approach},
  author={Sachdeva, Noveen and Dhaliwal, Mehak Preet and Wu, Carole-Jean and McAuley, Julian},
  booktitle={Advances in Neural Information Processing Systems},
  series={NeurIPS '22},
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

This repository already includes the pre-processed data for ML-1M, Amazon Magazine, and Douban datasets as described in the paper. The code for pre-processing is in `preprocess.py`.

---

## How to train ∞-AE?

- Edit the `hyper_params.py` file which lists all config parameters of ∞-AE.
- Finally, type the following command to train and evaluate ∞-AE:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

---

## Results sneak-peak

Below are the nDCG@10 results for the datasets used in the [paper](https://arxiv.org/abs/2206.02626):

| Dataset         | PopRec | MF    | NeuMF | MVAE  | LightGCN    | EASE  | ∞-AE      |
| ----------------- | -------- | ------- | ------- | ------- | ------------- | ------- | ------------ |
| Amazon Magazine | 8.42   | 13.1  | 13.6  | 12.18 | 22.57       | 22.84 | **23.06**  |
| MovieLens-1M    | 13.84  | 25.65 | 24.44 | 22.14 | 28.85       | 29.88 | **32.82**  |
| Douban          | 11.63  | 13.21 | 13.33 | 16.17 | 16.68       | 19.48 | **24.94**  |
| Netflix         | 12.34  | 12.04 | 11.48 | 20.85 | *Timed out* | 26.83 | **30.59*** |

*Note*: ∞-AE's results on the Netflix dataset (marked with a *) are obtained by training only on 5% of the total users. Note however, all other methods are trained on the *full* dataset.

---

## MIT License

