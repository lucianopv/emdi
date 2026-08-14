# The three jackknife MSE estimators in R/mse.R each rebuild the *in-sample*
# framework inside their delete-one-domain loop, even though it does not depend
# on the loop variable. framework_FH() runs makeXY()/model.matrix(), so this
# costs m redundant model-matrix builds per jackknife fit.
#
# These tests pin the call count, which is deterministic, rather than timing.

jk_combined <- local({
  load("FH/eusilcA_popAgg.RData")
  load("FH/eusilcA_smpAgg.RData")
  combine_data(
    pop_data = eusilcA_popAgg, pop_domains = "Domain",
    smp_data = eusilcA_smpAgg, smp_domains = "Domain"
  )
})

# Counts framework_FH() invocations during one fh() call.
count_framework_FH <- function(expr) {
  count <- 0L
  orig <- framework_FH
  fit <- testthat::with_mocked_bindings(
    force(expr),
    framework_FH = function(...) {
      count <<- count + 1L
      orig(...)
    },
    .package = "emdi"
  )
  list(fit = fit, count = count)
}

test_that("jiang_jackknife builds the in-sample framework once, not once per domain", {
  res <- count_framework_FH(
    suppressMessages(fh(
      fixed = MTMED ~ cash + age_ben + rent + house_allow,
      vardir = "Var_MTMED", combined_data = jk_combined, domains = "Domain",
      method = "ml", interval = c(0, 10000000),
      transformation = "arcsin", backtransformation = "naive",
      eff_smpsize = "n", MSE = TRUE, mse_type = "jackknife"
    ))
  )

  m <- res$fit$framework$N_dom_smp
  expect_gt(m, 1)
  # 1 framework for fh() itself + m delete-one frameworks + 1 in-sample framework
  expect_equal(res$count, m + 2L)
})

test_that("chen_weighted_jackknife builds the in-sample framework once, not once per domain", {
  res <- count_framework_FH(
    suppressMessages(fh(
      fixed = MTMED ~ cash + age_ben + rent + house_allow,
      vardir = "Var_MTMED", combined_data = jk_combined, domains = "Domain",
      method = "ml", interval = c(0, 10000000),
      transformation = "arcsin", backtransformation = "naive",
      eff_smpsize = "n", MSE = TRUE, mse_type = "weighted_jackknife"
    ))
  )

  m <- res$fit$framework$N_dom_smp
  expect_gt(m, 1)
  expect_equal(res$count, m + 2L)
})
