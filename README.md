# `fastrerandomize`: An R Package for Ultra-Fast Rerandomization Using Accelerated Computing 

[**What is `fastrerandomize`?**](#description)
| [**News**](#news)
| [**Performance**](#performance)
| [**Speed of Light Analysis**](#speed-of-light-analysis)
| [**Installation**](#installation)
| [**Tutorial**](#tutorial)
| [**Data**](#data)
| [**CRAN**](https://cran.r-project.org/web/packages/fastrerandomize/index.html)
| [**Documentation**](https://github.com/cjerzak/fastrerandomize-software/blob/main/fastrerandomize.pdf)
| [**References**](#references)

[<img src="https://img.shields.io/badge/Demo-View%20Demo-blue" alt="Demo Button">](https://cran.r-project.org/web/packages/fastrerandomize/vignettes/MainVignette.html) [<img src="https://img.shields.io/badge/CRAN-View%20on%20CRAN-green" alt="CRAN Button">](https://cran.r-project.org/web/packages/fastrerandomize/index.html) [<img src="https://img.shields.io/badge/Website-Visit%20Website-orange" alt="Website">](https://fastrerandomize.github.io)

<!---
[![Tests](https://github.com/cjerzak/fastrerandomize-software/actions/workflows/tests.yml/badge.svg)](https://github.com/cjerzak/fastrerandomize-software/actions/workflows/tests.yml)
--->

*Note:* `fastrerandomize` has been successfully tested on [CPU](https://en.wikipedia.org/wiki/Central_processing_unit), [CUDA](https://en.wikipedia.org/wiki/CUDA), [ROCm](https://en.wikipedia.org/wiki/ROCm), and  [METAL](https://en.wikipedia.org/wiki/Metal_(API)) frameworks. Special thanks to [Aniket Kamat](https://github.com/aniketkamat) and [Fucheng Warren Zhu](https://github.com/WarrenZhu050413) for their work on the latest package build! 

# What is `fastrerandomize`?<a id="description"></a>
`fastrerandomize` contains functions such as `randomization_test`, which offers a streamlined approach for performing randomization tests after using rerandomization in the research design. 

We employ a [JAX backend](https://en.wikipedia.org/wiki/Google_JAX) to make exact rerandomization inference possible even for larger experiments where the number of randomizations is in the hundreds of millions or where experimenters seek to maintain balanced randomizations across thousands of features.

# News<a id="news"></a>
**October 2026 (development version):** We profiled the package's accelerator kernels and found that most of each call's time went to overhead rather than computation. Every call ran a ~0.7 s conda check, Monte Carlo kernels were recompiled on every call, and the fiducial-interval search made about 1,000 small device round trips. We also found that int8 permutations could produce invalid treatment assignments on some GPU backends (observed on ROCm). The fixes:
- Monte Carlo generation now runs in a single compiled program, with random keys derived per draw. It returns the same randomizations as before for a given seed.
- Fiducial intervals use a closed form computed in one pass over the candidate randomizations.
- Balance statistics and distances use matrix-product formulations on covariates centered in double precision, which also improves float32 accuracy.
- Every generated assignment is now checked for validity.

On an AMD Radeon RX 7600M XT GPU, repeated calls became 10–110× faster for Monte Carlo generation, 6–180× for `randomization_test()`, 14× for exact enumeration, and 8–80× for `fast_distance()`. First calls also became faster. On CPU, `randomization_test()` with fiducial intervals went from 40 s to 0.35 s. See [NEWS.md](fastrerandomize/NEWS.md) for details.

# Performance: n = 100<a id="performance"></a>

<a href="https://arxiv.org/abs/2501.07642/#gh-light-mode-only">
  <img src="https://connorjerzak.com/wp-content/uploads/2026/01/CPU_v_GPU_FALSE_100_light.webp#gh-light-mode-only" alt="Figure – light" width="600">
</a>

<a href="https://arxiv.org/abs/2501.07642/#gh-dark-mode-only">
  <img src="https://connorjerzak.com/wp-content/uploads/2026/01/CPU_v_GPU_FALSE_100_dark.webp#gh-dark-mode-only" alt="Figure – dark" width="600">
</a>

# Performance: n = 1000

<a href="https://arxiv.org/abs/2501.07642/#gh-light-mode-only">
  <img src="https://connorjerzak.com/wp-content/uploads/2026/01/CPU_v_GPU_FALSE_1000_light.webp#gh-light-mode-only" alt="Figure – light" width="600">
</a>

<a href="https://arxiv.org/abs/2501.07642/#gh-dark-mode-only">
  <img src="https://connorjerzak.com/wp-content/uploads/2026/01/CPU_v_GPU_FALSE_1000_dark.webp#gh-dark-mode-only" alt="Figure – dark" width="600">
</a>

# Speed of Light Analysis<a id="speed-of-light-analysis"></a>

Skipping unused covariance preparation reduced the warmed complete R call from **26.42 s to 3.41 s (7.75× faster)** on an NVIDIA RTX 5080 Laptop GPU system using R’s reference BLAS. The benchmark used default diagonal scoring for 1,048,576 Monte Carlo draws with 4,096 units and 2,048 covariates (batch 4,096; 0.1% accepted), with identical assignments, random keys, and balance scores before and after. The GPU stage took 2.95 s, and its balance matrix multiplication sustained 18.6 TFLOP/s—about 53% of an analytical FP32 reference at NVIDIA’s listed upper boost clock. Permutation sorting remains the largest GPU cost. Hardware counters were unavailable, so the roofline percentage is an estimate.

[Read the full analysis](benchmarks/speed_of_light/REPORT.md).

# Package Installation and Loading <a id="installation"></a>
```
# Install fastrerandomize stable version from CRAN
# install.packages("fastrerandomize")

# Install fastrerandomize development version from GitHub
# devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")

# Load the package
library(  fastrerandomize  ) 

# Running code the first time, you'll want to create the computational environment 
# (detects Apple Silicon, NVIDIA, or AMD GPUs; use backend = "cpu" to skip the GPU)
fastrerandomize::build_backend()
```

# Tutorial<a id="tutorial"></a>
Let's get started with a tutorial. We're first going to use the package to generate a pool of acceptable rerandomizations.
```
# First, specify some analysis parameters
n_units <- 20; n_treated <- 10 

# Generate covariate data 
X <- matrix(rnorm(n_units*5),nrow = n_units)

# Generate a set of acceptable randomizations based on randomization_accept_prob.
# When randomization_accept_prob = 1, all randomizations are accepted. 
# When randomization_accept_prob < 1, only well-balanced randomizations are accepted. 
# When randomization_accept_prob = 1/|Size of cand. randomization set|, 1 randomization is accepted.
candidate_randomizations <- fastrerandomize::generate_randomizations(
                                            n_units = n_units,
                                            n_treated = n_treated,
                                            X = X,
                                            randomization_accept_prob = 0.001)

# The result is an S3 object with fields:
# - $randomizations: matrix of acceptable treatment assignments
# - $balance: corresponding balance statistics for each randomization

# Check out the candidate randomization dimensions
dim( candidate_randomizations$randomizations )
```
We can also use `fastrerandomize` to perform a randomization test using those acceptable randomizations. 
```
# Setup simulated outcome data 
CoefY <- rnorm(ncol(X))
Wobs <- candidate_randomizations$randomizations[1,]
tau_true <- 1
Yobs <- c(X %*% as.matrix(CoefY) + Wobs*tau_true + rnorm(n_units, sd = 0.1))

# Perform exact randomization set based on accepted randomizations 
ExactRandomizationTestResults <- fastrerandomize::randomization_test(
  obsW = Wobs,
  obsY = Yobs,
  candidate_randomizations = candidate_randomizations$randomizations,
  findFI = F # set to T if an exact fiducial interval needed
)
ExactRandomizationTestResults$p_value # p-value
ExactRandomizationTestResults$tau_obs # difference-in-means ATE estimate
```

# Replication Data<a id="data"></a>
Replication data for the package is available using the `data` command. 
```
data( QJEData )
data( YOPData )
```

# Development 
We welcome new features or bug fixes (you can raise an issue or submit a pull request in the repository). We will keep the package up-to-date with the latest version of the [JAX backend](https://en.wikipedia.org/wiki/Google_JAX). 

# References<a id="references"></a>
Connor T. Jerzak, Rebecca Goldstein, Aniket Kamat, Fucheng Warren Zhu. FastRerandomize: An R Package for Fast Rerandomization Using Accelerated Computing. *SoftwareX*, 2026. [[PDF]](https://arxiv.org/pdf/2501.07642)
```
@article{jerzak2025fastrerandomize,
  title={FastRerandomize: An R Package for Fast Rerandomization Using Accelerated Computing},
  author={Jerzak, Connor T. and Rebecca Goldstein and Aniket Kamat and Fucheng Warren Zhu},
  journal={SoftwareX},
  year={2026}
}
```


<!--
_

*Package functions* 
[<img src="https://github.com/cjerzak/fastrerandomize-software/blob/main/misc/figures/Viz_MainFxns.png?raw=true">](https://arxiv.org/pdf/2501.07642)

_

*Hardware acceleration* 
[<img src="https://github.com/cjerzak/fastrerandomize-software/blob/main/misc/figures/Viz_GPU.png?raw=true">](https://arxiv.org/pdf/2501.07642)

_

*Key to minimize memory overhead* 
[<img src="https://github.com/cjerzak/fastrerandomize-software/blob/main/misc/figures/Viz_Keys.png?raw=true">](https://arxiv.org/pdf/2501.07642)

_


[<img src="https://github.com/cjerzak/fastrerandomize-software/blob/main/misc/figures/CPU_v_GPU_FALSE_1000.png?raw=true">](https://arxiv.org/pdf/2501.07642)
-->
