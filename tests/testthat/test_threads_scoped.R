# The kernels take an explicit thread count and apply it with a num_threads()
# clause scoped to their own parallel region. All RNG is pre-generated before
# every parallel region, so results must be bit-identical at any thread count.

test_that("monte_carlo_cpp accepts a threads argument and is invariant to it", {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")
  framework <- framework_ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, custom_indicator = NULL,
    na.rm = TRUE, pop_weights = NULL, weights = NULL
  )
  set.seed(42)
  pe <- point_estim(framework = framework, fixed = eqIncome ~ gender + eqsize,
                    transformation = "log", interval = "default", L = 20)

  run <- function(n) {
    set.seed(7)
    point_estim(framework = framework, fixed = eqIncome ~ gender + eqsize,
                transformation = "log", interval = "default", L = 20,
                threads = n)$point_estimates
  }
  expect_equal(run(1L), run(4L), tolerance = 1e-12)
})

# The value assertions above cannot catch a pragma that silently drops
# num_threads(threads): the RNG is pre-generated, so results agree at any
# thread count regardless. This checks the structural property directly.
test_that("every OpenMP parallel region takes its thread count from an argument", {
  src_dir <- testthat::test_path("..", "..", "src")
  skip_if_not(dir.exists(src_dir), "source tree not available (installed package)")

  for (f in c("monte_carlo.cpp", "parametric_bootstrap.cpp",
              "fh_arcsin.cpp", "fh_jackknife.cpp")) {
    p <- file.path(src_dir, f)
    skip_if_not(file.exists(p), paste(f, "not found"))
    directives <- grep("pragma omp parallel", readLines(p), value = TRUE)
    expect_gt(length(directives), 0)
    expect_true(
      all(grepl("num_threads(threads)", directives, fixed = TRUE)),
      info = paste(f, "has an omp parallel region without num_threads(threads):",
                   paste(directives, collapse = " | "))
    )
  }
})
