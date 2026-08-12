# The jackknife and spatial-bootstrap loops in R/mse.R emitted one
# newline-terminated message per iteration ("domain =1\n", "b =1\n", ...),
# unconditionally. At B = 500 or m = 500 that is 500 lines of scrollback and
# 500 stderr flushes.
#
# The package's established idiom for iteration progress is a single in-place
# line updated with a leading carriage return (see R/mse_estimation.R and
# src/parametric_bootstrap.cpp). These tests pin that: progress must not grow
# one line per iteration.

prog_combined <- local({
  load("FH/eusilcA_popAgg.RData")
  load("FH/eusilcA_smpAgg.RData")
  combine_data(
    pop_data = eusilcA_popAgg, pop_domains = "Domain",
    smp_data = eusilcA_smpAgg, smp_domains = "Domain"
  )
})

test_that("jackknife progress does not emit one line per domain", {
  # In-sample domains: those carrying a direct estimate / sampling variance.
  m <- sum(!is.na(prog_combined$Var_MTMED))
  expect_gt(m, 1)

  msgs <- testthat::capture_messages(
    fh(
      fixed = MTMED ~ cash + age_ben + rent + house_allow,
      vardir = "Var_MTMED", combined_data = prog_combined, domains = "Domain",
      method = "ml", interval = c(0, 10000000),
      transformation = "arcsin", backtransformation = "naive",
      eff_smpsize = "n", MSE = TRUE, mse_type = "jackknife"
    )
  )

  # The old form. Nothing should match it.
  expect_length(grep("^domain =", msgs), 0)

  # Progress output must not grow one line per domain: in a terminal it is
  # overwritten in place, and when redirected to a log it is throttled.
  expect_lt(sum(grepl("\n$", msgs)), m)
})

test_that("spatial parametric bootstrap progress does not emit one line per iteration", {
  skip_on_cran()
  # The 94-domain package fixtures: eusilcA_prox is 94 x 94, so the small
  # 15-domain FH/*.RData fixtures used above cannot be paired with it.
  data("eusilcA_popAgg"); data("eusilcA_smpAgg"); data("eusilcA_prox")
  sp_combined <- combine_data(
    eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain"
  )

  msgs <- testthat::capture_messages(
    fh(
      fixed = Mean ~ cash + self_empl,
      vardir = "Var_Mean", combined_data = sp_combined, domains = "Domain",
      method = "reml", correlation = "spatial",
      corMatrix = as.matrix(eusilcA_prox),
      MSE = TRUE, mse_type = "spatialparboot", B = 10
    )
  )

  expect_length(grep("^b =", msgs), 0)
  # Fewer progress lines than the B = 10 bootstrap iterations that produced them.
  expect_lt(sum(grepl("\n$", msgs)), 10)
})
