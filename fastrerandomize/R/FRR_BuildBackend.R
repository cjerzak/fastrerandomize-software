#' A function to build the environment for fastrerandomize. Builds a conda environment in which 'JAX' and 'np' are installed. Users can also create a conda environment where 'JAX' and 'np' are installed themselves.
#'
#' @param conda_env (default = `"fastrerandomize_env"`) Name of the conda environment in which to place the backends.
#' @param conda (default = `auto`) The path to a conda executable. Using `"auto"` allows reticulate to attempt to automatically find an appropriate conda binary.
#' @param backend (default = `"auto"`) The accelerator backend to install. `"auto"` picks
#'   `"metal"` on Apple Silicon, `"cuda"` on Linux with an NVIDIA driver, `"rocm"` on Linux
#'   with an AMD GPU, and `"cpu"` otherwise. Set it explicitly to override detection, for
#'   example `backend = "cpu"` on a GPU machine.
#'
#' @details
#' On AMD GPUs (`"rocm"`), the ROCm runtime is installed into the environment from AMD's
#' Python wheel index (about 3 GB), so only the amdgpu kernel driver is needed. GPUs that
#' 'JAX' does not compile for, but that run the code of a supported GPU (the RX 7600 series,
#' gfx1102), are mapped to it with `HSA_OVERRIDE_GFX_VERSION_<node>`, and other AMD GPUs
#' (such as an integrated GPU) are hidden with `ROCR_VISIBLE_DEVICES`. These variables are
#' stored in the environment (`conda env config vars`) and applied when the package loads
#' 'JAX'. Sorting on some ROCm GPUs is not reproducible, which `generate_randomizations()`
#' reports as an error; use `backend = "cpu"` (or set `JAX_PLATFORMS=cpu`) if this occurs.
#'
#' The function ends by reporting the device 'JAX' uses in the new environment.

#' @return Invisibly returns NULL; this function is used for its side effects
#' of creating and configuring a conda environment for `fastrerandomize`.
#' This function requires an Internet connection.
#' You can find out a list of conda Python paths via: `Sys.which("python")`
#'
#' @examples
#' \dontrun{
#' # Create a conda environment named "fastrerandomize_env"
#' # and install the required Python packages (jax, numpy, etc.)
#' build_backend(conda_env = "fastrerandomize_env", conda = "auto")
#'
#' # If you want to specify a particular conda path:
#' # build_backend(conda_env = "fastrerandomize_env", conda = "/usr/local/bin/conda")
#'
#' # Install CPU-only JAX on a machine with a GPU:
#' # build_backend(backend = "cpu")
#' }
#'
#' @export
#' @md

build_backend <- function(conda_env = "fastrerandomize_env", conda = "auto",
                          backend = c("auto", "cpu", "cuda", "rocm", "metal")){
  backend <- match.arg(backend)
  os <- Sys.info()[["sysname"]]
  machine <- Sys.info()[["machine"]]
  msg <- function(...) message(sprintf(...))

  # Resolve the backend before creating anything
  drv <- if (identical(os, "Linux")) .nvidia_driver() else NA_character_
  drv_major <- suppressWarnings(as.integer(sub("^([0-9]+).*", "\\1", drv)))
  amd_gpus <- .amd_gpus()
  backend <- .resolve_backend(backend, os, machine, drv_major, amd_gpus)

  # Create a new conda environment
  reticulate::conda_create(envname = conda_env,
                           conda = conda,
                           python_version = "3.13")
  env_python <- reticulate::conda_python(envname = conda_env, conda = conda)

  pip_install <- function(pkgs, ...) {
    reticulate::py_install(packages = pkgs, envname = conda_env, conda = conda, pip = TRUE, ...)
    TRUE
  }

  # Install numpy first (pinned version)
  pip_install("numpy")

  # --- Install JAX with appropriate backend ---
  env_vars <- character()
  if (backend == "metal") {
    msg("Apple Silicon detected: installing JAX 0.5.0 with Metal support.")
    pip_install(c("jax==0.5.0", "jaxlib==0.5.0", "jax-metal==0.1.1"))
  } else if (backend == "cuda") {
    # Prefer CUDA 13 if the driver is new enough; otherwise CUDA 12
    if (!is.na(drv_major) && drv_major >= 580) {
      msg("Driver %s detected (>=580): installing JAX CUDA 13 wheels.", drv)
      tryCatch(pip_install('jax[cuda13]'), error = function(e) {
        msg("CUDA 13 wheels failed (%s); falling back to CUDA 12.", e$message)
        pip_install('jax[cuda12]')
      })
    } else {
      msg("Driver %s: installing JAX CUDA 12 wheels.", if (is.na(drv)) "not detected" else drv)
      pip_install('jax[cuda12]')
    }
  } else if (backend == "rocm") {
    rocm <- .rocm_config(amd_gpus)
    msg("AMD GPU %s detected: installing JAX with ROCm (%s device libraries).",
        rocm$gfx, rocm$wheel_gfx)
    .install_jax_rocm(pip_install, env_python, rocm$wheel_gfx)
    env_vars <- rocm$env_vars
    .set_conda_env_vars(conda, env_python, env_vars)
  } else {
    if (identical(os, "Linux") && !is.na(drv_major)) {
      msg("Driver %s is too old for CUDA wheels (needs >= 525).", drv)
    }
    msg("Installing CPU-only JAX.")
    pip_install('jax')
  }

  # (Optional) neutralize LD_LIBRARY_PATH inside this env to prevent overrides
  if (os == "Linux") {
    try({
      # Python path is typically: <env_path>/bin/python, so go up two directories
      env_path <- dirname(dirname(env_python))
      actdir <- file.path(env_path, "etc", "conda", "activate.d")
      dir.create(actdir, recursive = TRUE, showWarnings = FALSE)
      writeLines("unset LD_LIBRARY_PATH", file.path(actdir, "00-unset-ld.sh"))
    }, silent = TRUE)
  }

  device <- .backend_smoke_test(env_python, env_vars)
  msg("JAX device in '%s': %s", conda_env, device)
  if (backend %in% c("cuda", "rocm") && !grepl("cuda|rocm|gpu", device, ignore.case = TRUE)) {
    msg(paste0("JAX did not find the GPU. ",
               if (backend == "rocm") sprintf(paste0(
                 "If JAX does not support %s, setting HSA_OVERRIDE_GFX_VERSION_<node> to a ",
                 "supported GPU of the same family may help. "), rocm$gfx),
               "Rebuild with backend = \"cpu\" to install CPU-only JAX."))
  }

  msg("Environment '%s' is ready.", conda_env)
  invisible(NULL)
}

.nvidia_driver <- function() {
  if (!nzchar(Sys.which("nvidia-smi"))) { return(NA_character_) }
  drv <- tryCatch(suppressWarnings(system2(
    "nvidia-smi", c("--query-gpu=driver_version", "--format=csv,noheader"),
    stdout = TRUE, stderr = FALSE)), error = function(e) character())
  if (length(drv)) trimws(drv[1L]) else NA_character_
}

.resolve_backend <- function(backend, os, machine, drv_major, amd_gpus) {
  linux <- identical(os, "Linux")
  apple_silicon <- identical(os, "Darwin") && identical(machine, "arm64")
  if (backend == "auto") {
    if (apple_silicon) { return("metal") }
    if (linux && !is.na(drv_major) && drv_major >= 525) { return("cuda") }
    if (linux && nrow(amd_gpus) > 0L) { return("rocm") }
    return("cpu")
  }
  problem <- switch(backend,
    metal = if (!apple_silicon) "requires macOS on Apple Silicon",
    cuda = if (!linux) "requires Linux",
    rocm = if (!linux) "requires Linux" else if (nrow(amd_gpus) == 0L)
      "found no AMD GPU in /sys/class/kfd (is the amdgpu driver loaded?)",
    NULL)
  if (!is.null(problem)) {
    stop(sprintf("backend = \"%s\" %s.", backend, problem), call. = FALSE)
  }
  backend
}

# AMD GPUs listed by the amdgpu kernel driver. Each node's gfx_target_version
# is major * 10000 + minor * 100 + stepping (0 for CPU nodes); node is the
# index used by HSA_OVERRIDE_GFX_VERSION_<node>, and ordinal the index used by
# ROCR_VISIBLE_DEVICES.
.amd_gpus <- function(topology = "/sys/class/kfd/kfd/topology/nodes") {
  nodes <- suppressWarnings(as.integer(list.files(topology)))
  nodes <- sort(nodes[!is.na(nodes)])
  prop <- function(node, key) {
    lines <- tryCatch(readLines(file.path(topology, node, "properties"), warn = FALSE),
                      error = function(e) character())
    hit <- grep(sprintf("^%s\\s", key), lines, value = TRUE)
    if (length(hit)) as.numeric(sub(".*\\s", "", trimws(hit[1L]))) else 0
  }
  version <- vapply(nodes, prop, numeric(1), key = "gfx_target_version")
  simd <- vapply(nodes, prop, numeric(1), key = "simd_count")
  gpu <- version > 0
  version <- version[gpu]
  data.frame(node = nodes[gpu],
             ordinal = seq_len(sum(gpu)) - 1L,
             gfx = sprintf("gfx%d%x%x", version %/% 10000,
                           (version %/% 100) %% 100, version %% 100),
             simd_count = simd[gpu])
}

# GPUs that JAX does not compile for, mapped to a supported GPU whose code they
# run unchanged (gfx1102 runs gfx1100 code; gfx1103 code lacks a gfx1102
# hardware workaround, so it is not a substitute).
.rocm_gfx_override <- list(gfx1102 = c(gfx = "gfx1100", version = "11.0.0"))

# Use the largest AMD GPU and any others of the same type: hide the rest and
# override unsupported types per node (a global override also applies to, and
# can break, an integrated GPU of another generation).
.rocm_config <- function(amd_gpus) {
  gfx <- amd_gpus$gfx[which.max(amd_gpus$simd_count)]
  use <- amd_gpus[amd_gpus$gfx == gfx, , drop = FALSE]
  env_vars <- character()
  if (nrow(use) < nrow(amd_gpus)) {
    env_vars["ROCR_VISIBLE_DEVICES"] <- paste(use$ordinal, collapse = ",")
  }
  override <- .rocm_gfx_override[[gfx]]
  if (!is.null(override)) {
    env_vars[sprintf("HSA_OVERRIDE_GFX_VERSION_%d", use$node)] <- override[["version"]]
  }
  list(gfx = gfx,
       wheel_gfx = if (is.null(override)) gfx else override[["gfx"]],
       env_vars = env_vars)
}

# JAX's ROCm plugin is released after JAX itself and does not pin it, so match
# jax/jaxlib to the plugin. The ROCm runtime comes only from AMD's index: PyPI
# has unrelated placeholder packages with the same names.
.install_jax_rocm <- function(pip_install, env_python, wheel_gfx) {
  pip_install("jax-rocm10-plugin")
  version <- system2(env_python, c("-c", shQuote(paste0(
    "import importlib.metadata as m; print(m.version('jax-rocm10-plugin'))"))),
    stdout = TRUE)
  version <- sub("\\+.*$", "", trimws(version[length(version)]))
  pip_install(sprintf(c("jax==%s", "jaxlib==%s"), version))
  pip_install(sprintf("rocm[libraries,device-%s]==10.*", wheel_gfx),
              pip_options = "--index-url https://stable.repo.amd.com/rocm/whl-next/")
}

.set_conda_env_vars <- function(conda, env_python, env_vars) {
  if (!length(env_vars)) { return(invisible(FALSE)) }
  status <- tryCatch(system2(
    reticulate::conda_binary(conda),
    c("env", "config", "vars", "set", "-p", shQuote(dirname(dirname(env_python))),
      shQuote(paste0(names(env_vars), "=", env_vars)))),
    error = function(e) 1L)
  if (!identical(as.integer(status), 0L)) {
    warning(sprintf("Could not store %s in the conda environment; set them before loading JAX.",
                    paste0(names(env_vars), "=", env_vars, collapse = " ")), call. = FALSE)
  }
  invisible(identical(as.integer(status), 0L))
}

.backend_smoke_test <- function(env_python, env_vars) {
  out <- tryCatch(suppressWarnings(system2(
    env_python, c("-c", shQuote("import jax; print(jax.devices()[0])")),
    stdout = TRUE, stderr = FALSE, timeout = 300,
    env = if (length(env_vars)) paste0(names(env_vars), "=", env_vars) else character())),
    error = function(e) character())
  if (length(out)) out[length(out)] else "none (JAX failed to load)"
}
