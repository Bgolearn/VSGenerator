

<table>
  <tr>
    <td width="160" align="center" valign="top">
      <img src="https://github.com/user-attachments/assets/7e77bd5a-42d6-45db-b8e6-2c82cac81b9d" width="140" style="border-radius: 50%;"/>
    </td>
    <td valign="top">
      <b>For any inquiries or assistance, feel free to contact Mr. CAO Bin at:</b><br>
      📧 Email: <a href="mailto:bcao686@connect.hkust-gz.edu.cn">bcao686@connect.hkust-gz.edu.cn</a><br><br>
      Cao Bin is a PhD candidate at the <b>Hong Kong University of Science and Technology (Guangzhou)</b>, 
      under the supervision of Professor <a href="https://gbaaa.org.hk/en-us/article/67">Zhang Tong-Yi</a>. His research focuses on 
      <b>AI for science</b>, especially intelligent crystal-structure analysis and discovery. 
      Learn more about his work on his 
      <a href="https://www.caobin.asia/">homepage</a>.
    </td>
  </tr>
</table>

---

**Associated publication**
[Science Bulletin](https://www.sciencedirect.com/science/article/pii/S2095927325012678)

# VSGenerator

**DVSNet: Dynamic Virtual Space Generation Neural Network** [![PyPI Version](https://img.shields.io/pypi/v/VSGenerator.svg)](https://pypi.org/project/VSGenerator/)

## Scientific Motivation

In data-driven materials discovery and other AI-for-science applications,
multi-objective optimization problems often involve **strongly imbalanced evaluation costs**.
Some target properties can be measured or labeled at low cost, while others require
expensive experiments, long simulations, or destructive characterization.

Conventional Bayesian optimization or active-learning strategies typically assume
uniform evaluation costs and therefore suffer from severe inefficiency when applied
to such asymmetric settings. This limitation becomes a critical bottleneck in
closed-loop experimental design.

## Method Overview

`VSGenerator` introduces **DVSNet**, a *conditional variational autoencoder*
designed to learn **adaptive virtual design spaces** from partially labeled data.

The core idea is to decouple multi-objective optimization into sequential stages:

1. **Low-cost objective optimization**
   The objective that is inexpensive to evaluate is optimized first, producing a set
   of feasible or high-performing candidates.

2. **Adaptive virtual space construction**
   DVSNet is trained to learn a conditional latent representation that captures the
   design subspace consistent with the constraints imposed by the first objective.

3. **Constrained high-cost optimization**
   The expensive objective is optimized exclusively within the generated virtual space,
   thereby avoiding unnecessary labeling for each sample.

This strategy enables **cost-aware exploration**, substantially reducing the number of
high-cost evaluations required to identify optimal solutions.

## Key Features

* Conditional variational autoencoder for adaptive design-space generation
* Explicit support for multi-objective optimization with non-uniform evaluation costs
* Seamless integration with the [Bgolearn](https://github.com/Bin-Cao/Bgolearn) framework
* Suitable for closed-loop experimental design and data-efficient materials discovery

## Installation

Install the package via pip:

```bash
pip install VSGenerator
```

## Usage and Reproducibility

A complete, reproducible workflow is provided in the tutorial notebook:

[https://github.com/Bgolearn/VSGenerator/blob/main/tutorial/VSGenerator_Tutorial.ipynb](https://github.com/Bgolearn/VSGenerator/blob/main/tutorial/VSGenerator_Tutorial.ipynb)

The tutorial demonstrates:

* Model initialization and training of DVSNet
* Construction of adaptive virtual spaces
* Integration with downstream optimization pipelines

## License

This project is distributed under the **MIT License**.
See the [LICENSE](LICENSE) file for licensing details.

---

## Citation

If you use `VSGenerator` or DVSNet in your research, please cite the associated paper:

```
@article{Cao2025SpatialAdaptiveAL,
  title   = {Spatial-adaptive active learning identifies ultra-durable and highly active catalysts for acidic oxygen evolution reaction},
  author  = {Cao, Bin and
             Qin, Yin and
             Luo, Yan and
             Ying, Zhehan and
             Yan, Zilin and
             Weng, Lu-Tao and
             Li, Kaikai and
             Zhang, Tong-Yi},
  journal = {Science Bulletin},
  year    = {2025},
  doi     = {10.1016/j.scib.2025.12.021},
  url     = {https://www.sciencedirect.com/science/article/pii/S2095927325012678?via%3Dihub#m0005}
}


```



