# EBP honours the engine switch.
#
# Until now `engine = "r"` was impossible for EBP: monte_carlo() called
# monte_carlo_cpp() directly, the pure-R Monte-Carlo loop having been replaced
# rather than kept. Upstream still maintains that implementation, so it is
# restored here as the "r" branch -- which is also why keeping it costs little:
# it is upstream's code, and staying mergeable means carrying it anyway.

ebp_engine_data <- function() {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")
  list(smp = eusilcA_smp, pop = eusilcA_pop)
}

run_ebp_engine <- function(engine, L = 8, seed = 5, ...) {
  d <- ebp_engine_data()
  withr::with_options(list(emdi.engine = engine), {
    set.seed(seed)
    suppressMessages(ebp(
      fixed = eqIncome ~ gender + eqsize,
      pop_data = d$pop, pop_domains = "district",
      smp_data = d$smp, smp_domains = "district",
      threshold = 10924.32, transformation = "log",
      L = L, MSE = FALSE, ...
    ))
  })
}

test_that("engine = 'r' does not reach the Monte-Carlo kernel", {
  skip_on_cran()
  called <- FALSE
  orig <- monte_carlo_cpp
  testthat::with_mocked_bindings(
    invisible(run_ebp_engine("r")),
    monte_carlo_cpp = function(...) { called <<- TRUE; orig(...) },
    .package = "emdi2"
  )
  expect_false(called)
})

test_that("engine = 'cpp' does reach the Monte-Carlo kernel", {
  skip_on_cran()
  called <- FALSE
  orig <- monte_carlo_cpp
  testthat::with_mocked_bindings(
    invisible(run_ebp_engine("cpp")),
    monte_carlo_cpp = function(...) { called <<- TRUE; orig(...) },
    .package = "emdi2"
  )
  expect_true(called)
})

test_that("both engines produce the same ebp() point estimates", {
  skip_on_cran()
  r <- run_ebp_engine("r")$ind
  cpp <- run_ebp_engine("cpp")$ind

  expect_equal(names(r), names(cpp))
  expect_equal(nrow(r), nrow(cpp))
  # The two implementations consume the RNG stream in the same order, so this
  # is an exact comparison rather than a statistical one. If that ever ceases
  # to hold, this assertion is the thing that says so.
  expect_equal(cpp$Mean, r$Mean, tolerance = 1e-8)
  expect_equal(cpp$Head_Count, r$Head_Count, tolerance = 1e-8)
  expect_equal(cpp$Gini, r$Gini, tolerance = 1e-8)
})
