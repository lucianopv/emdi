# C++ support for a user-supplied `true_indicators`.
#
# A supplied truth is constant across all B iterations, so the kernel reads it
# once and skips its Step 2. That skip is not itself worth much -- Step 5's
# Monte-Carlo loop does the same indicator work L times per iteration -- but a
# supplied truth previously forced the whole bootstrap onto the R loop.
# Measured on eusilcA at L = 20, B = 20: 5.76s before, 4.22s now (~1.35x),
# against 4.00s for the ordinary C++ path.
#
# Value-identity between the engines is not enough on its own here: the R and
# C++ bootstraps draw independent RNG streams, so they cannot be compared
# directly. These tests therefore check (a) that the C++ path is actually taken,
# via a dispatch probe, and (b) that the supplied truth genuinely reaches the
# kernel and changes the answer, by supplying two different truths.

ti_data <- function() {
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")
  list(smp = eusilcA_smp, pop = eusilcA_pop)
}

# A truth frame covering every population domain, in any order.
make_truth <- function(pop, value) {
  doms <- unique(pop$district)
  all_names <- c("Mean", "Head_Count", "Poverty_Gap", "Gini",
                 "Quintile_Share", "Quantile_10", "Quantile_25",
                 "Median", "Quantile_75", "Quantile_90")
  df <- data.frame(Domain = doms, stringsAsFactors = FALSE)
  for (nm in all_names) df[[nm]] <- value
  df
}

run_ebp <- function(truth, B = 3, ...) {
  d <- ti_data()
  set.seed(1)
  suppressMessages(ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = d$pop, pop_domains = "district",
    smp_data = d$smp, smp_domains = "district",
    threshold = 10924.32, transformation = "log",
    L = 5, MSE = TRUE, B = B, true_indicators = truth, ...
  ))
}

test_that("a supplied true_indicators now takes the C++ path", {
  skip_on_cran()
  d <- ti_data()
  called <- FALSE
  orig <- parametric_bootstrap_cpp   # capture BEFORE mocking, or this recurses

  testthat::with_mocked_bindings(
    invisible(run_ebp(make_truth(d$pop, 1000))),
    parametric_bootstrap_cpp = function(...) { called <<- TRUE; orig(...) },
    .package = "emdi"
  )
  expect_true(called)
})

test_that("the supplied truth reaches the kernel and changes the MSE", {
  skip_on_cran()
  d <- ti_data()
  # MSE is mean over B of (estimate - truth)^2, so a truth far from the
  # estimates must give a larger MSE than one close to them. If the kernel
  # ignored the argument, both runs would return the same numbers.
  near <- run_ebp(make_truth(d$pop, 1e4))$MSE$Mean
  far  <- run_ebp(make_truth(d$pop, 1e6))$MSE$Mean

  expect_true(all(is.finite(near)))
  expect_true(all(is.finite(far)))
  expect_gt(mean(far), mean(near))
})

test_that("supplied truth works at the aggregate_to level", {
  skip_on_cran()
  d <- ti_data()
  d$pop$region <- as.integer(as.factor(d$pop$district)) %% 5 + 1

  doms <- unique(d$pop$region)
  all_names <- c("Mean", "Head_Count", "Poverty_Gap", "Gini",
                 "Quintile_Share", "Quantile_10", "Quantile_25",
                 "Median", "Quantile_75", "Quantile_90")
  truth <- data.frame(Domain = doms)
  for (nm in all_names) truth[[nm]] <- 1e4

  set.seed(1)
  res <- suppressMessages(ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = d$pop, pop_domains = "district",
    smp_data = d$smp, smp_domains = "district",
    threshold = 10924.32, transformation = "log",
    L = 5, MSE = TRUE, B = 3,
    aggregate_to = "region", true_indicators = truth
  ))

  expect_equal(nrow(res$MSE), length(doms))
  expect_true(all(is.finite(res$MSE$Mean)))
})

test_that("a truth at the wrong level is rejected rather than misaligned", {
  skip_on_cran()
  d <- ti_data()
  d$pop$region <- as.integer(as.factor(d$pop$district)) %% 5 + 1

  # district-level truth supplied while aggregating to region
  expect_error(
    suppressMessages(ebp(
      fixed = eqIncome ~ gender + eqsize,
      pop_data = d$pop, pop_domains = "district",
      smp_data = d$smp, smp_domains = "district",
      threshold = 10924.32, transformation = "log",
      L = 5, MSE = TRUE, B = 2,
      aggregate_to = "region", true_indicators = make_truth(d$pop, 1e4)
    )),
    "aggregate_to"
  )
})

test_that("only the MSE_indicators columns need supplying", {
  skip_on_cran()
  d <- ti_data()
  truth <- make_truth(d$pop, 1e4)[, c("Domain", "Mean", "Head_Count")]

  set.seed(1)
  res <- suppressMessages(ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = d$pop, pop_domains = "district",
    smp_data = d$smp, smp_domains = "district",
    threshold = 10924.32, transformation = "log",
    L = 5, MSE = TRUE, B = 3,
    MSE_indicators = c("Mean", "Head_Count"), true_indicators = truth
  ))
  expect_true(all(is.finite(res$MSE$Mean)))
  expect_true(all(is.finite(res$MSE$Head_Count)))
})
