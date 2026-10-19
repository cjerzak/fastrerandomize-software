# Backend detection for build_backend() (pure R, no JAX required)

fake_topology <- function(nodes) {
  root <- tempfile("kfd")
  for (i in seq_along(nodes)) {
    dir.create(file.path(root, names(nodes)[i]), recursive = TRUE)
    writeLines(c("cpu_cores_count 0",
                 sprintf("simd_count %d", nodes[[i]][["simd"]]),
                 "mem_banks_count 1",
                 sprintf("gfx_target_version %d", nodes[[i]][["gfx"]])),
               file.path(root, names(nodes)[i], "properties"))
  }
  root
}

# A CPU node, an RX 7600M XT (gfx1102) and an integrated gfx1036, as on a laptop
laptop <- fake_topology(list(`0` = c(simd = 0, gfx = 0),
                             `1` = c(simd = 64, gfx = 110002),
                             `2` = c(simd = 4, gfx = 100306)))

test_that(".amd_gpus parses the KFD topology", {
  gpus <- fastrerandomize:::.amd_gpus(laptop)
  expect_equal(gpus$node, c(1L, 2L))
  expect_equal(gpus$ordinal, c(0L, 1L))
  expect_equal(gpus$gfx, c("gfx1102", "gfx1036"))
  expect_equal(gpus$simd_count, c(64, 4))
})

test_that(".amd_gpus is empty without the amdgpu driver", {
  gpus <- fastrerandomize:::.amd_gpus(tempfile("missing"))
  expect_equal(nrow(gpus), 0L)
  expect_equal(nrow(fastrerandomize:::.amd_gpus(
    fake_topology(list(`0` = c(simd = 0, gfx = 0))))), 0L)
})

test_that(".amd_gpus names targets with hexadecimal steppings", {
  root <- fake_topology(list(`0` = c(simd = 0, gfx = 0),
                             `2` = c(simd = 440, gfx = 90010),
                             `10` = c(simd = 1216, gfx = 90402),
                             `11` = c(simd = 128, gfx = 120001)))
  gpus <- fastrerandomize:::.amd_gpus(root)
  expect_equal(gpus$gfx, c("gfx90a", "gfx942", "gfx1201"))
  expect_equal(gpus$node, c(2L, 10L, 11L))
})

test_that(".rocm_config overrides gfx1102 per node and hides the integrated GPU", {
  rocm <- fastrerandomize:::.rocm_config(fastrerandomize:::.amd_gpus(laptop))
  expect_equal(rocm$gfx, "gfx1102")
  expect_equal(rocm$wheel_gfx, "gfx1100")
  expect_equal(rocm$env_vars, c(ROCR_VISIBLE_DEVICES = "0",
                                HSA_OVERRIDE_GFX_VERSION_1 = "11.0.0"))
})

test_that(".rocm_config leaves supported GPUs alone", {
  one <- fake_topology(list(`0` = c(simd = 0, gfx = 0),
                            `1` = c(simd = 96, gfx = 110000)))
  rocm <- fastrerandomize:::.rocm_config(fastrerandomize:::.amd_gpus(one))
  expect_equal(rocm$wheel_gfx, "gfx1100")
  expect_length(rocm$env_vars, 0L)

  two <- fake_topology(list(`0` = c(simd = 0, gfx = 0),
                            `1` = c(simd = 96, gfx = 110000),
                            `2` = c(simd = 96, gfx = 110000)))
  expect_length(fastrerandomize:::.rocm_config(fastrerandomize:::.amd_gpus(two))$env_vars, 0L)
})

test_that(".resolve_backend picks and validates backends", {
  resolve <- fastrerandomize:::.resolve_backend
  amd <- fastrerandomize:::.amd_gpus(laptop)
  none <- fastrerandomize:::.amd_gpus(tempfile("missing"))

  expect_equal(resolve("auto", "Darwin", "arm64", NA, none), "metal")
  expect_equal(resolve("auto", "Darwin", "x86_64", NA, none), "cpu")
  expect_equal(resolve("auto", "Linux", "x86_64", 580L, amd), "cuda")
  expect_equal(resolve("auto", "Linux", "x86_64", 470L, none), "cpu")
  expect_equal(resolve("auto", "Linux", "x86_64", NA, amd), "rocm")
  expect_equal(resolve("auto", "Linux", "x86_64", NA, none), "cpu")
  expect_equal(resolve("auto", "Windows", "x86-64", NA, none), "cpu")

  expect_equal(resolve("cpu", "Linux", "x86_64", 580L, amd), "cpu")
  expect_equal(resolve("cuda", "Linux", "x86_64", NA, none), "cuda")
  expect_equal(resolve("rocm", "Linux", "x86_64", 580L, amd), "rocm")
  expect_error(resolve("rocm", "Linux", "x86_64", NA, none), "no AMD GPU")
  expect_error(resolve("rocm", "Darwin", "arm64", NA, none), "requires Linux")
  expect_error(resolve("cuda", "Darwin", "arm64", NA, none), "requires Linux")
  expect_error(resolve("metal", "Linux", "x86_64", NA, none), "Apple Silicon")
})
