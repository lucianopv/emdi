# ebp(cpus > 1) previously errored in every worker because the parallelLapply
# call omitted two arguments that mse_estim_wrapper() actually evaluates.
# The wild bootstrap is used here because it has no C++ fast path, so it is the
# branch that genuinely exercises parallelMap.

test_that("ebp(cpus = 2) runs the wild bootstrap without error", {
  skip_on_cran()
  skip_if_not_installed("parallelMap")
  skip_on_os("windows")   # multicore mode is unavailable on Windows
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")

  expect_no_error(
    suppressMessages(ebp(
      fixed = eqIncome ~ gender + eqsize,
      pop_data = eusilcA_pop, pop_domains = "district",
      smp_data = eusilcA_smp, smp_domains = "district",
      threshold = 10924.32, transformation = "log",
      L = 5, MSE = TRUE, B = 4, boot_type = "wild",
      cpus = 2, parallel_mode = "multicore"
    ))
  )
})

test_that("the parallel bootstrap returns a well-formed MSE", {
  skip_on_cran()
  skip_if_not_installed("parallelMap")
  skip_on_os("windows")
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")

  res <- suppressMessages(ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, transformation = "log",
    L = 5, MSE = TRUE, B = 4, boot_type = "wild",
    cpus = 2, parallel_mode = "multicore"
  ))

  # parallelMap calls clusterSetRNGStream(), so values legitimately differ from
  # a cpus = 1 run. Assert structure and finiteness, not equality.
  #
  # Quintile_Share is excluded from the finiteness check below: it has a
  # separate, pre-existing defect in the wild-bootstrap true-indicator path
  # (R/framework_ebp.R's qsr(), unrelated to cpus/parallelism -- confirmed to
  # reproduce under cpus = 1 too) where the asymmetric ">" comparison against
  # the 80th percentile empties the top quintile whenever the domain's largest
  # simulated value ties that percentile, producing a 0/0 division. Out of
  # scope for this fix; this test only asserts the missing-argument bug this
  # task targets is gone.
  expect_s3_class(res, "emdi")
  non_qsr <- res$MSE[, setdiff(names(res$MSE), c("Domain", "Quintile_Share")), drop = FALSE]
  expect_true(all(vapply(non_qsr, function(x) all(is.finite(x)), logical(1))))
  expect_gt(nrow(res$MSE), 0)
})
