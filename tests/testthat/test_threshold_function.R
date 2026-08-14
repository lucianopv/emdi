# ebp() accepts a function-valued `threshold` (a relative poverty line). The
# C++ bootstrap kernel takes a single scalar threshold for all B iterations,
# whereas mse_estim() re-evaluates the closure against each replicate's own
# superpopulation -- so the fast path is declined for a closure rather than
# resolved once, which would silently change the statistic.
#
# Before this was gated, the closure was passed straight to the kernel and
# ebp() died with:
#   Not compatible with requested type: [type=closure; target=double]
# The package's own compare_plot.Rd example triggers exactly this.

test_that("uses_cpp_bootstrap declines the fast path for a function threshold", {
  args <- list(boot_type = "parametric", n_indicators = 10L,
               true_indicators = NULL)
  expect_true(do.call(uses_cpp_bootstrap, c(args, list(threshold = 10924.32))))
  expect_false(do.call(uses_cpp_bootstrap,
                       c(args, list(threshold = function(y) 0.6 * median(y)))))
  # absent threshold must not accidentally disable the fast path
  expect_true(do.call(uses_cpp_bootstrap, args))
})

test_that("ebp(MSE = TRUE) works with a function-valued threshold", {
  skip_on_cran()
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")

  set.seed(1)
  expect_no_error(
    res <- suppressMessages(ebp(
      fixed = eqIncome ~ gender + eqsize,
      pop_data = eusilcA_pop, pop_domains = "district",
      smp_data = eusilcA_smp, smp_domains = "district",
      threshold = function(y) 0.6 * median(y),
      transformation = "log", L = 5, MSE = TRUE, B = 3
    ))
  )
  expect_true(all(is.finite(res$MSE$Mean)))
  expect_true(all(is.finite(res$MSE$Head_Count)))
})
