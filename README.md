# Normalizing Flows & GLOW: Generative Image Modeling

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()
[![Topic](https://img.shields.io/badge/Topic-Generative_Models_%26_Deep_Learning-purple.svg)]()

An implementation and exploration of **Normalizing Flows** for generative image modeling, with a focus on the **RNVP (Real-valued Non-Volume Preserving)** architecture. The project applies flow-based generative models to digit image generation and investigates the latent space structure of a facial attribute dataset.

---

## 📌 Project Overview

Normalizing flows learn exact likelihood generative models by transforming a simple base distribution (e.g., Gaussian) into a complex data distribution through a sequence of invertible, differentiable transformations.

### Key Areas Covered

1. **RNVP Architecture**: Implementation of Real-valued Non-Volume Preserving flows for density estimation and generation of handwritten digits.
2. **Image Generation**: Sampling from the learned distribution to produce new realistic digit images.
3. **Latent Space Exploration**: Traversing the latent space of a facial attribute dataset (Smiling / Not Smiling) to study semantic interpolation.
4. **GLOW Model**: Study and application of the Generative Flow with Invertible 1×1 Convolutions (GLOW) model.

---

## 📂 Repository Structure

```directory
glow-normalizing-flows/
├── Lab9_PaulaE_JoelB.ipynb    # Main notebook: theory, implementation & experiments
├── rnvp-digits.pt              # Pre-trained RNVP model checkpoint (digit generation)
└── 02-NormalizingFlows.pdf     # Reference lecture slides on Normalizing Flows
```

---

## 🧪 Methodology

- **Base Model**: RNVP with affine coupling layers trained on digit images.
- **Architecture**: Stacked invertible coupling layers with alternating masking patterns.
- **Training**: Exact maximum likelihood via change of variables formula: $\log p(x) = \log p_z(f(x)) + \log \left| \det \frac{\partial f}{\partial x} \right|$.
- **Facial Dataset**: CelebA-style dataset used to probe latent disentanglement of facial attributes.

---

## 🔬 Key Concepts

| Concept | Description |
| :--- | :--- |
| **Normalizing Flow** | Sequence of invertible transformations mapping $p_z \to p_x$ |
| **Affine Coupling Layer** | Core RNVP building block: $y_{1:d} = x_{1:d}$, $y_{d+1:D} = x_{d+1:D} \odot \exp(s(x_{1:d})) + t(x_{1:d})$ |
| **Exact Likelihood** | Unlike VAEs/GANs, flows provide exact log-likelihood computation |
| **1×1 Convolution (GLOW)** | Learnable permutation layer generalizing channel shuffling |

---

## 🚀 How to Run

Open the notebook in Jupyter or Google Colab (GPU recommended):

```bash
jupyter notebook Lab9_PaulaE_JoelB.ipynb
```

To load the pre-trained RNVP model:

```python
import torch
model = torch.load('rnvp-digits.pt')
model.eval()
samples = model.sample(n=16)
```

---

## 👥 Authors & License

Developed as part of the **Advanced Probabilistic and Neural Systems (APRNS)** course at GIA.  
Distributed under the **MIT License**.
