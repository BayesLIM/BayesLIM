(install)=

# Installation

Clone the GitHub repo: <https://github.com/nkern/BayesLIM>
and then 

```bash
cd BayesLIM
pip install .
```

If installed properly, you should be able to import it in a Python session as:

```python
import bayeslim as ba
```

## Dependencies

See the [`pyproject.toml`](https://github.com/nkern/BayesLIM/blob/main/pyproject.toml) file for dependencies, listed under `[project.optional-dependencies]`. I placed them here so that they won't automatically install during a `pip install .` by default, which doesn't always play nicely with software sharing similar dependencies. If you'd like to have pip install the dependencies anyways, you can make this happen by installing the `[dev]` version:

```bash
pip install .[dev]
```

**PyTorch**: to my knowledge there is not a great way to install different CPU/GPU versions of PyTorch from within a `pyproject.toml`, but I suspect this will change relatively soon. In the meantime, you should install `torch>=2.0.0` (and optionally CUDA if you want GPU support) on your own (<https://pytorch.org/get-started/locally/>) before installing BayesLIM.