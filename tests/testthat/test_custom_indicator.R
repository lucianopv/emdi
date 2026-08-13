# custom_indicator with MSE = TRUE used to fail with
#   object 'f' of mode 'function' was not found
#
# Cause: n_std is 10, the number of standard indicator NAMES, but
# framework$indicator_list holds FUNCTIONS -- only 6, because `quants` alone
# produces 5 of the 10 names. Slicing [(n_std + 1):length(indicator_list)]
# built a DESCENDING index (11:7 with one custom indicator), so most entries
# were NULL from out-of-bounds positions. match.fun(NULL) is not a function, so
# it fell back to resolving the symbol `f`, which does not exist there.
#
# Custom functions are appended after the standard ones, so the slice must take
# the last n_custom entries. These tests exercise one and several custom
# indicators, which is what makes a descending-index regression visible.

ci_data <- function() {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")
  list(smp = eusilcA_smp, pop = eusilcA_pop)
}

run_ci <- function(ci, MSE = TRUE, B = 3) {
  d <- ci_data()
  set.seed(1)
  suppressMessages(ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = d$pop, pop_domains = "district",
    smp_data = d$smp, smp_domains = "district",
    threshold = 10924.32, transformation = "log",
    L = 5, MSE = MSE, B = B, custom_indicator = ci
  ))
}

test_that("one custom indicator works with MSE = TRUE", {
  skip_on_cran()
  res <- run_ci(list(my_max = function(y, pop_weights, threshold) max(y)))
  expect_true("my_max" %in% names(res$ind))
  expect_true(all(is.finite(res$ind$my_max)))
  expect_true("my_max" %in% names(res$MSE))
})

test_that("several custom indicators work with MSE = TRUE", {
  skip_on_cran()
  # More than one is what distinguishes a correct slice from a descending one:
  # with n_custom = 3 the old code produced indices 11:7 regardless.
  res <- run_ci(list(
    my_max  = function(y, pop_weights, threshold) max(y),
    my_min  = function(y, pop_weights, threshold) min(y),
    my_sd   = function(y, pop_weights, threshold) stats::sd(y)
  ))
  for (nm in c("my_max", "my_min", "my_sd")) {
    expect_true(nm %in% names(res$ind), info = nm)
    expect_true(all(is.finite(res$ind[[nm]])), info = nm)
  }
  # Each custom indicator must get its OWN values, not a recycled neighbour's.
  expect_false(isTRUE(all.equal(res$ind$my_max, res$ind$my_min)))
  expect_gt(min(res$ind$my_max - res$ind$my_min), 0)
})

test_that("custom indicators still work with MSE = FALSE", {
  skip_on_cran()
  res <- run_ci(list(my_max = function(y, pop_weights, threshold) max(y)),
                MSE = FALSE)
  expect_true(all(is.finite(res$ind$my_max)))
})
