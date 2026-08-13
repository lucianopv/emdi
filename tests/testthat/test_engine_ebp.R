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

# --- the lambda search --------------------------------------------------------
# optimal_parameter() is the second EBP path that lost its R branch: our version
# calls optimal_parameter_cpp() unconditionally. Upstream's optimize(generic_opt,
# ...) call is restorable in full, and cheaply -- generic_opt itself never left
# this package (plot.ebp.R still calls it to draw the lambda profile).

run_ebp_bc <- function(engine, L = 8, seed = 5) {
  d <- ebp_engine_data()
  withr::with_options(list(emdi.engine = engine), {
    set.seed(seed)
    suppressMessages(ebp(
      fixed = eqIncome ~ gender + eqsize,
      pop_data = d$pop, pop_domains = "district",
      smp_data = d$smp, smp_domains = "district",
      threshold = 10924.32, transformation = "box.cox",
      L = L, MSE = FALSE
    ))
  })
}

test_that("engine = 'r' does not reach the lambda-search kernel", {
  skip_on_cran()
  called <- FALSE
  orig <- optimal_parameter_cpp
  testthat::with_mocked_bindings(
    invisible(run_ebp_bc("r")),
    optimal_parameter_cpp = function(...) { called <<- TRUE; orig(...) },
    .package = "emdi2"
  )
  expect_false(called)
})

test_that("engine = 'cpp' does reach the lambda-search kernel", {
  skip_on_cran()
  called <- FALSE
  orig <- optimal_parameter_cpp
  testthat::with_mocked_bindings(
    invisible(run_ebp_bc("cpp")),
    optimal_parameter_cpp = function(...) { called <<- TRUE; orig(...) },
    .package = "emdi2"
  )
  expect_true(called)
})

test_that("both engines find the same box.cox optimum", {
  skip_on_cran()
  # The two lambdas differ by ~4.2e-06, and that is expected: upstream leaves
  # optimize() at its default tol = .Machine$double.eps^0.25 = 1.22e-04, so the
  # R path stops short of the optimum the C++ Brent reaches. Measured on this
  # fixture: R gives 0.363952125, cpp 0.363956291. Re-running optimize() on the
  # same objective with tol = 1e-6 or 1e-9 moves it to 0.36395635 -- i.e. onto
  # the C++ answer -- which is what identifies the cause as the optimizer's
  # stopping rule and not a difference in the likelihood being profiled.
  #
  # So the bar is optimize()'s own convergence tolerance. Asserting anything
  # tighter would be asserting that R's default optimizer is more precise than
  # it claims to be.
  lam_r <- run_ebp_bc("r")$transform_param$optimal_lambda
  lam_cpp <- run_ebp_bc("cpp")$transform_param$optimal_lambda
  expect_equal(lam_cpp, lam_r, tolerance = .Machine$double.eps^0.25)

  # The invariant that actually matters: both sit on the same flat optimum of
  # the same REML likelihood. Objective difference is 1.4e-08 out of 2.0e+04,
  # a relative 7e-13 -- and the C++ value is the lower of the two.
  obj <- function(lam) {
    generic_opt(lam, eqIncome ~ gender + eqsize, ebp_engine_data()$smp,
                "district", "box.cox", nlme::lmeControl(opt = "optim"))
  }
  expect_equal(obj(lam_cpp), obj(lam_r), tolerance = 1e-9)
  expect_lte(obj(lam_cpp), obj(lam_r))
})

# --- the MSE bootstrap --------------------------------------------------------
# This one needs no restoration: mse_estim() kept its R loop as the fallback for
# wild bootstrap, custom indicators and function-valued thresholds. The switch
# only has to reach the existing gate.

test_that("uses_cpp_bootstrap declines the fast path under engine = 'r'", {
  args <- list(boot_type = "parametric", n_indicators = 10L,
               true_indicators = NULL, threshold = 10924.32)
  expect_false(withr::with_options(list(emdi.engine = "r"),
                                   do.call(uses_cpp_bootstrap, args)))
  expect_true(withr::with_options(list(emdi.engine = "cpp"),
                                  do.call(uses_cpp_bootstrap, args)))
})
