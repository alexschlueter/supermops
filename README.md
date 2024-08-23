# SuperMoPS - **Super**-resolution of **Mo**ving **P**oint **S**ources
This repository contains the code for the paper **"Dimension reduction, exact recovery, and error estimates for sparse reconstruction in phase space"** by M. Holler, A. Schlüter and B. Wirth ([arXiv](https://arxiv.org/abs/2112.09743), [ACHA](https://doi.org/10.1016/j.acha.2024.101631)). If you use this code, please cite the paper as reference.

### Abstract
An important theme in modern inverse problems is the reconstruction of *time-dependent* data from only *finitely many* measurements. To obtain satisfactory reconstruction results in this setting it is essential to strongly exploit temporal consistency between the different measurement times. The strongest consistency can be achieved by reconstructing data directly in *phase space*, the space of positions *and* velocities. However, this space is usually too high-dimensional for feasible computations. We introduce a novel dimension reduction technique, based on projections of phase space onto lower-dimensional subspaces, which provably circumvents this curse of dimensionality: Indeed, in the exemplary framework of superresolution we prove that known exact reconstruction results stay true after dimension reduction, and we additionally prove new error estimates of reconstructions from noisy data in optimal transport metrics which are of the same quality as one would obtain in the non-dimension-reduced case.

## Installation
First, clone the repository and its submodule:
```
git clone --recurse-submodules https://github.com/alexschlueter/supermops.git
cd supermops
```

To install the `supermops` package, run
```
pip install .
```
in the root directory of this repository. If you want be able to change the code in the supermops directory and see the changes reflected in the installed version immediately, use `pip install -e .` instead to install the package in editable mode.

For additional packages needed to run the provided scripts, run:
```
pip install -r requirements.txt
```

If incompatibilities with some packages arise, you can also try the known working versions in `requirements_versions.txt`, which have been tested to work with Python 3.11.2.

To solve the resulting optimization problems, we call the commercial MOSEK solver throught the CVXPY library. For this, you need to install a MOSEK license in the appropriate path, [see here for academic licenses](https://www.mosek.com/products/academic-licenses/).
If you really want to circumvent this, change the lines where the solver is called to `solver.solve(use_mosek=False)`. You may also need to uninstall the mosek package again with `pip uninstall mosek` to prevent CVXPY from calling it. Note however that this is untested and the results might be inconsistent with the data from the article.

If you are only interested in trying our method, the above installation is all that is needed.
However, for comparison, we also ran experiments with the ADCG algorithm applied to the full-dimensional dynamic formulation of the superresolution problem. For this, we used the Julia implementation of Alberti et al. [2], which is contained in the git submodule "dynamic_spike_super_resolution" in this repository (with some modifications to work for higher Julia versions and to allow for the resumption of reconstructions).
If you want to run these experiments as well, you need to install Julia (the scripts were tested with Julia v1.9.2) and prepare the package using
```
julia --project=./dynamic_spike_super_resolution -e"using Pkg; Pkg.instantiate();"
```

## Running

For a simple, commented example showcasing the main functions, check out the file `example.py` and run it using
```
python example.py
```

## Math summary
Let $\Omega\subset\mathbb{R}^d$ be a compact domain and denote by $\mathcal{M}(X)$ the space of Radon measures on a domain $X$ and by $\lVert\cdot\rVert_\mathcal{M}$ the total variation norm on Radon measures.
Following Alberti et al. [2], we model a collection of $N$ linearly moving particles with positions $x_i$ at time $t=0$ and velocities $v_i$ as a linear combination $`\lambda^\dagger={\sum}_{i=1}^N m_i \delta_{(x_i,v_i)}\in\mathcal{M}(\mathbb{R}^d\times\mathbb{R}^d)`$ of Dirac measures, where $m_i\geq 0$ are some weights. Using the notation $`{g}_\#\nu`$ for the pushforward of a measure $\nu$ under a function $g$, we define the **Move operator** at a time $t\in\mathbb{R}$ as

```math
 \mathrm{Mv} ^d_t:\mathcal{M}(\mathbb{R}^d\times\mathbb{R}^d)\to\mathcal{M}(\mathbb{R}^d),\, \mathrm{Mv} ^d_t\lambda={[(x,v)\mapsto x+tv]}_\#\lambda.
```

Applying this to the phase space measure $`\lambda^\dagger={\sum}_{i=1}^N m_i \delta_{(x_i,v_i)}`$ results in the **snapshot** at time t,

```math
u_t^\dagger \coloneqq  \mathrm{Mv} ^d_t\lambda^\dagger = \sum_{i=1}^N m_i \delta_{x_i+tv_i}.
```

We assume that, at each time step $t$ from a finite set of measurement times ${\mathcal T}\subset\mathbb{R}$, some data $f_t^\dagger=\mathrm{Ob}_t u_t^\dagger$ measured with observation operators $\mathrm{Ob}_t\colon\mathcal{M}(\Omega)\to H$ in a Hilbert space $H$ is available.
We compare three different convex optimization problems in order to reconstruct the particle configuration:

First, the static problem at a time instant $t$, which tries to reconstruct the snapshot $`u_t^\dagger={\sum}_{i=1}^N m_i\delta_{x_i+tv_i}`$ using only the data $f_t^\dagger$ from this time instant:
```math
    \min_{u\in\mathcal{M}_+(\Omega)}\lVert u\rVert_\mathcal{M}
    \quad\text{such that } \mathrm{Ob}_tu=f^\dagger_t
```
Second, the full-dimensional dynamic problem proposed by Alberti et al. in [2], which is set in an appropriate subset $\Lambda\subset\mathbb{R}^d\times\mathbb{R}^d$ of  $2d$-dimensional phase space and tries to reconstruct $\lambda^\dagger$ directly:
```math
\min_{\lambda \in \mathcal{M}_+(\Lambda)} \| \lambda \|_\mathcal{M} \quad \text{such that } \mathrm{Ob}_t \mathrm{Mv} ^d_t \lambda =  f_t^\dagger \quad \text{for all }t \in {\mathcal T}.
```

Third, our new proposed dimension-reduced version of the above problem. To define the dimension reduction, for a unit vector $\theta\in S^{d-1}$, we introduce the **Radon transform** and the **joint Radon transform** for measures as

```math
\begin{align*}
 \mathrm{Rd} _{\theta}&:\mathcal{M}(\mathbb{R}^d)\to\mathcal{M}(\mathbb{R}),&& \mathrm{Rd} _\theta\nu={[x\mapsto\theta\cdot x]}_\#\nu,\\
 \mathrm{Rj} _{\theta}&:\mathcal{M}(\mathbb{R}^d\times\mathbb{R}^d)\to\mathcal{M}(\mathbb{R}^2),&& \mathrm{Rj} _\theta\lambda={[(x,v)\mapsto(\theta\cdot x,\theta\cdot v)]}_\#\lambda.
\end{align*}
```

The new dimension-reduced variable $\gamma$ is the **position-velocity projection** resulting from applying the joint Radon transform to $`\lambda^\dagger={\sum}_{i=1}^N m_i \delta_{(x_i,v_i)}`$ for some $\theta\in S^{d-1}$:

```math
\gamma_\theta^\dagger\coloneqq \mathrm{Rj} _\theta\lambda^\dagger=\sum_{i=1}^N m_i \delta_{(\theta\cdot x_i,\theta\cdot v_i)}\in\mathcal{M}(\mathbb{R}^2)
```

Taking an appropriate domain $\Gamma\subset\mathbb{R}^2$, subsets $\Sigma\subset\mathbb{R}$ and $\Theta\subset S^{d-1}$, our dimension-reduced problem now reads

```math
\begin{equation*}
\min_{ \substack{ \gamma \in \mathcal{M}_+(\Theta \times \Gamma) \\ u \in \mathcal{M}_+(\Omega)^{\Sigma} }}
\| \gamma \|_\mathcal{M} \quad \text{such that }
\begin{cases}
 \mathrm{Mv} _t^1 \gamma_\theta =  \mathrm{Rd} _\theta u_t &\text{for all }t \in \Sigma,\theta\in\Theta, \\
\mathrm{Ob}_tu_t = f_t^\dagger &\text{for all }t \in {\mathcal T}.
\end{cases}
\end{equation*}
```

The first constraint is sometimes referred to as **projected / reduced time consistency** in the code, while the second constraint ensures consistency with the measured data.

## Notes about the implementation
There are some differences between the article and the code, some due to historical differences in variable names and some to allow for a simpler implementation.

**First, the position-velocity projection variable $\gamma$ is called $\mu$ (`mu`) in this code.**

Second, in the implementation of our optimization problem, the **snapshots $u_t$ do not occur directly**, but rather a transformed version called $\nu$ (`nu`) is used. The relationship between the variables is the following: For some $n\in S^1$, define the **reparametrized Move operator** as

```math
 \mathrm{rMv} ^d_n:\mathcal{M}(\mathbb{R}^d\times\mathbb{R}^d)\to\mathcal{M}(\mathbb{R}^d),\, \mathrm{rMv} ^d_n\lambda={[(x,v)\mapsto n_1x+n_2v]}_\#\lambda.
```

Now, $\nu_n$ is a measure in $\mathcal{M}(\mathbb{R}^d)$ thought to represent $` \mathrm{rMv} ^d_n\lambda`$ for a phase space variable $\lambda\in\mathcal{M}(\mathbb{R}^d\times\mathbb{R}^d)$. If $t\in\Sigma$ is a time step, define the corresponding direction $n_t\in S^1$ as $n_t\coloneqq (1, t) / \sqrt{1 + t^2}$. Then we have

```math
\begin{align*}
\nu_{n_t} =  \mathrm{rMv} ^d_{n_t}\lambda &= {[x\mapsto x\cdot 1/\sqrt{1+t^2}]}_\#{[(x,v)\mapsto x+tv]}_\#\lambda\\&={[x\mapsto x\cdot 1/\sqrt{1+t^2}]}_\#u_t,
\end{align*}
```

so $\nu_{n_t}$ is just a rescaling of the domain of the snapshot $u_t$. The numerical problem we solve is now a discretization of

```math
\begin{equation*}
\begin{gathered}
\min_{\substack{\mu\in\mathcal{M}(\mathbb{R}^2)^\Theta\\ \nu\in\mathcal{M}(\mathbb{R}^d)^\mathcal{N}}} \sum_{\theta\in\Theta}\lVert\mu_\theta\rVert_\mathcal{M} + \sum_{n\in\mathcal{N}}\lVert\nu_n\rVert_\mathcal{M} + \frac{1}{2\alpha}\sum_{t\in{\mathcal T}}\lVert \mathrm{Ob}_t {[x\mapsto\sqrt{1+t^2}x]}_\#\nu_{n_t} - f_t\rVert_H^2\\
    \text{s.t.} \left(\sum_{\theta\in\Theta,n\in\mathcal{N}}\lVert  \mathrm{rMv} ^1_n\mu_\theta -  \mathrm{Rd} _\theta\nu_n\rVert_2^2\right)^{1/2} \leq \tau
\end{gathered}
\end{equation*}
```

for some subset of directions $\Theta\subset S^{d-1}$ called `mu_dirs`, some subset $\mathcal{N}\subset S^1$ called `nu_dirs`, and parameters $\alpha$ (`paper_alpha`) and $\tau$ (`redcons_bound`). Note that the 2-norm in the constraint only makes sense after discretization. Also, the reparametrized Move operation in one dimension, which occurs in the constraint, is the same as the Radon transform in 2D, so the same discretized implementation can be used for both.
See `supermops/solvers/cvxpy_solver.py` for the actual implementation of the optimization problem.

Although many parts of the code are generalized for arbitrary space dimensions $d$, box domains $\Omega$=`main_bbox` and measurement times, the only well-tested code paths are for $d=2$, `main_bbox = BoundingBox([[0, 1], [0, 1]])` and times in the form `np.linspace(-1, 1, 2*K+1)`.
Especially the stripe projection only works for $d=2$ and the Fourier measurement operator relies on `main_bbox` being the unit cube.


## Experiment pipeline

The pipeline to run experiments consists of the following steps, using files contained in the `scripts/` directory:
1. Generate a dataset of ground truth particle configurations
2. Define an experiment by writing a `params.py`
3. Generate a `params.json` from the python file
4. Run `reconstruct_v7.py` for our model or `reconstruct_adcg_v2.jl` for ADCG
5. Run `evaluate_v6.py` to evaluate each reconstruction result
6. Run `collect_eval_v2.py` to collect all evaluation results into a single `eval_all.h5` file
7. Load `eval_all.h5` using the `supermops.eval.PDEval` class, analyse and plot the results contained in the pandas DataFrames

If the dataset in step 1 is supposed to have uniform distribution of dynamic separation values (as generated by `gen_ground_truth.py`), we first need to generate a distribution of separation values to initialize the rejection-sampling algorithm (this is heavily inspired by the implementation of [2]). This is done by running `gen_separation_distr.py` and afterwards `collect_separations.py`.

### Job parameters
Each experiment consists of many sets of parameters called "pjobs". These are written to the `params.json` and processed during the next steps of the pipeline.
The reconstruction and evaluation steps were run for many pjobs in parallel on a compute cluster and will probably take too long to run on a single machine. The scripts have some common parameters to control the splitting of pjobs into groups for easier parallelization, see for example the output of `python reconstruct_v7.py --help`. The two most important parameters are
```
--reduction N # split the pjobs contained in the experiment into groups of N
--jobid j # in the current invocation, process all pjobs from the j-th group of the split
```

## How to reproduce the paper
All evaluation results (i.e. the `eval_all.h5` files from step 6 for each experiment) as well as the datasets of ground truth particle configurations can be downloaded to the folder `article/data` by running
```
python scripts/download_article_data.py
```

(or manually downloaded from http://doi.org/10.17879/09978425554).

If you simply want to explore the raw evaluation data, open `article/notebooks/explore_article_eval.ipynb` with jupyter.
To recreate the plots in the article from the provided data, enter the subdirectories in `article/plots` and run `make` (you need a latex installation for this).


For each step in the pipeline, we now describe how to reproduce the results from the paper. We also provide a `scripts/reproduce_paper.py` file, which should in principle run these steps automatically. Note however, that this is only provided for demonstration purposes, as the actual data for the paper was produced on an HPC cluster and reconstructing all examples on a single machine would take much too long.
1. Download the ground truth datasets as described above. Set the `REDLIN_GT_ROOT` environment variable in your current shell to the absolute path of the `article/data/ground_truth` folder.
2. The `params.py` files for all experiments are provided in the `article/experiments` folder.

For each experiment (i.e. for each separate `params.py`), run the commands for the following steps in the experiment root folder, which is the folder containing the `params.py`. Also, let `$git` be the path to this git repo.

3. Run `python params.py --write-json` resulting in a `params.json` file
4. If the experiment is an ADCG experiment (check if the method field in the json is "ADCG"): run

```
julia --project=$git/dynamic_spike_super_resolution $git/scripts/reconstruct_adcg_v2.jl --param-json params.json
```

(with appropriate job parameters, see previous section). If the experiment has "PyRedLin" in the method fields, run
```
python $git/scripts/reconstruct_v7.py --param-json params.json
```
instead.

5. Run `python $git/scripts/evaluate_v6.py --param-json params.json` (with appropriate job parameters)
6. Run `python $git/scripts/collect_eval_v2.py --param-json params.json`

You should now have one `eval_all.h5` per experiment containing the collected evaluation results.

## References

[1] [Dimension reduction, exact recovery, and error estimates for sparse reconstruction in phase space](https://arxiv.org/abs/2112.09743) by Martin Holler, Alexander Schlüter and Benedikt Wirth (2021)

[2] [Dynamic Spike Superresolution and Applications to Ultrafast Ultrasound Imaging](https://epubs.siam.org/doi/10.1137/18M1174775) by Giovanni S. Alberti, Habib Ammari, Francisco Romero, and Timothée Wintz (2019)

[3] [Dimension reduction, exact recovery, and error estimates for sparse reconstruction in phase space](https://doi.org/10.1016/j.acha.2024.101631) by M. Holler, A. Schlüter, B. Wirth, Applied and Computational Harmonic Analysis (2024)

