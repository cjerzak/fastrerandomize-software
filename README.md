# `fastrerandomize`: An R Package for Ultra-Fast Rerandomization Using Accelerated Computing 

[**What is `fastrerandomize`?**](#description)
| [**Performance**](#performance)
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

*Note:* `fastrerandomize` has been successfully tested on [CPU](https://en.wikipedia.org/wiki/Central_processing_unit), [CUDA](https://en.wikipedia.org/wiki/CUDA), and [METAL](https://en.wikipedia.org/wiki/Metal_(API)) frameworks. Special thanks to [Aniket Kamat](https://github.com/aniketkamat) and [Fucheng Warren Zhu](https://github.com/WarrenZhu050413) for their work on the latest package build! 

# What is `fastrerandomize`?<a id="description"></a>
`fastrerandomize` contains functions such as `randomization_test`, which offers a streamlined approach for performing randomization tests after using rerandomization in the research design. 

We employ a [JAX backend](https://en.wikipedia.org/wiki/Google_JAX) to make exact rerandomization inference possible even for larger experiments where the number of randomizations is in the hundreds of millions or where experimenters seek to maintain balanced randomizations across thousands of features.

# Performance: n = 100<a id="performance"></a>

<a href="https://arxiv.org/abs/2501.07642/#gh-light-mode-only">
  <img src="https://connorjerzak.com/wp-content/uploads/2025/11/figure1.webp#gh-light-mode-only" alt="Figure – light" width="600">
</a>

<a href="https://arxiv.org/abs/2501.07642/#gh-dark-mode-only">
  <img src="https://connorjerzak.com/wp-content/uploads/2025/11/figure1_dark.webp#gh-dark-mode-only" alt="Figure – dark" width="600">
</a>

# Performance: n = 1000

<a href="https://arxiv.org/abs/2501.07642/#gh-light-mode-only">
  <img src="https://connorjerzak.com/wp-content/uploads/2025/11/figure2.webp#gh-light-mode-only" alt="Figure – light" width="600">
</a>

<a href="https://arxiv.org/abs/2501.07642/#gh-dark-mode-only">
  <img src="https://connorjerzak.com/wp-content/uploads/2025/11/figure2_dark.webp#gh-dark-mode-only" alt="Figure – dark" width="600">
</a>

# Package Installation and Loading <a id="installation"></a>
```
# Install fastrerandomize stable version from CRAN
# install.packages("fastrerandomize")

# Install fastrerandomize development version from GitHub
# devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")

# Load the package
library(  fastrerandomize  ) 

# Running code the first time, you'll want to create the computational environment 
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

# Check out the candidate randomization dimensions 
dim( candidate_randomizations )
```
We can also use `fastrerandomize` to perform a randomization test using those acceptable randomizations. 
```
# Setup simulated outcome data 
CoefY <- rnorm(ncol(X))
Wobs <- candidate_randomizations[1,]
tau_true <- 1
Yobs <- c(X %*% as.matrix(CoefY) + Wobs*tau_true + rnorm(n_units, sd = 0.1))

# Perform exact randomization set based on accepted randomizations 
ExactRandomizationTestResults <- fastrerandomize::randomization_test(
  obsW = Wobs,
  obsY = Yobs,
  candidate_randomizations = candidate_randomizations,
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
Connor T. Jerzak, Rebecca Goldstein, Aniket Kamat, Fucheng Warren Zhu. fastrerandomize: An R Package for Fast Rerandomization Using Accelerated Computing. *ArXiv Preprint*, 2025. [[PDF]](https://arxiv.org/pdf/2501.07642)
```
@article{jerzak2025fastrerandomize,
  title={fastrerandomize: An R Package for Fast Rerandomization Using Accelerated Computing},
  author={Jerzak, Connor T. and Rebecca Goldstein and Aniket Kamat and Fucheng Warren Zhu},
  journal={ArXiv Preprint},
  year={2025}
}
```

Connor T. Jerzak and Rebecca Goldstein. "Degrees of Randomness in Rerandomization Procedures." *ArXiv Preprint*, 2023. [\[PDF\]](https://arxiv.org/pdf/2310.00861.pdf)
```
@article{JerGol2023,
         title={Degrees of Randomness in Rerandomization Procedures},
         author={Jerzak, Connor T. and Rebecca Goldstein},
         journal={ArXiv Preprint},
         year={2023}}
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
