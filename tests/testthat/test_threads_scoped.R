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
