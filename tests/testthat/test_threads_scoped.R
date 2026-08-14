# The kernels take an explicit thread count and apply it with a num_threads()
# clause scoped to their own parallel region. All RNG is pre-generated before
# every parallel region, so results must be bit-identical at any thread count.

test_that("monte_carlo_cpp accepts a threads argument and is invariant to it", {
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")
  framework <- framework_ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, custom_indicator = NULL,
    na.rm = TRUE, pop_weights = NULL, weights = NULL
  )
  nthr <- emdi_cores(4L)
  skip_if(nthr < 2L, "needs at least 2 cores to be meaningful")

  run <- function(n) {
    set.seed(7)
    point_estim(framework = framework, fixed = eqIncome ~ gender + eqsize,
                transformation = "log", interval = "default", L = 20,
                threads = n)$point_estimates
  }
  expect_equal(run(1L), run(nthr), tolerance = 1e-12)
})

# The value assertions above cannot catch a pragma that silently drops
# num_threads(threads): the RNG is pre-generated, so results agree at any
# thread count regardless. This checks the structural property directly.
test_that("every OpenMP parallel region takes its thread count from an argument", {
  src_dir <- testthat::test_path("..", "..", "src")
  skip_if_not(dir.exists(src_dir), "source tree not available (installed package)")

  cpp_files <- list.files(src_dir, pattern = "\\.cpp$", full.names = TRUE)
  skip_if(length(cpp_files) == 0, "no .cpp files found under src/")

  all_directives <- character(0)
  for (p in cpp_files) {
    directives <- grep("pragma omp parallel", readLines(p), value = TRUE)
    if (length(directives) == 0) next   # e.g. RcppExports.cpp has no OpenMP regions
    expect_true(
      all(grepl("num_threads(threads)", directives, fixed = TRUE)),
      info = paste(basename(p), "has an omp parallel region without num_threads(threads):",
                   paste(directives, collapse = " | "))
    )
    all_directives <- c(all_directives, directives)
  }
  # A fifth source file with an OpenMP region and no num_threads(threads) would
  # be caught above; this catches the file list itself silently going empty.
  expect_gte(length(all_directives), 4L)
})

# ---------------------------------------------------------------------------
# cpus is a core budget, and it buys exactly one kind of parallelism: OpenMP
# threads on the C++ path, worker processes on the R fallback -- never both.
# ---------------------------------------------------------------------------

test_that("ebp() uses the C++ bootstrap even when cpus > 1", {
  skip_on_cran()
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")
  # The C++ path never starts parallelMap workers. If cpus > 1 still forced the
  # R fallback, parallelStart would be called.
  called <- FALSE
  orig <- parametric_bootstrap_cpp   # capture BEFORE mocking, or this recurses
  testthat::with_mocked_bindings(
    {
      set.seed(1)
      invisible(suppressMessages(ebp(
        fixed = eqIncome ~ gender + eqsize,
        pop_data = eusilcA_pop, pop_domains = "district",
        smp_data = eusilcA_smp, smp_domains = "district",
        threshold = 10924.32, transformation = "log",
        L = 5, MSE = TRUE, B = 3, cpus = 2)))
    },
    parametric_bootstrap_cpp = function(...) { called <<- TRUE; orig(...) },
    .package = "emdi"
  )
  expect_true(called)
})

test_that("ebp() bootstrap MSE is identical at cpus 1 and 2 on the C++ path", {
  skip_on_cran()
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")
  run <- function(n) {
    set.seed(1)
    suppressMessages(ebp(
      fixed = eqIncome ~ gender + eqsize,
      pop_data = eusilcA_pop, pop_domains = "district",
      smp_data = eusilcA_smp, smp_domains = "district",
      threshold = 10924.32, transformation = "log",
      L = 5, MSE = TRUE, B = 3, cpus = n))$MSE
  }
  expect_equal(run(1L), run(2L), tolerance = 1e-12)
})

# The two blocks above prove the fast path stays on, but not that the budget
# actually arrives at the kernels: both would still pass if some hop silently
# defaulted threads back to 1. These two read the value the kernel was handed.
test_that("ebp() hands the resolved budget to the Monte-Carlo kernel", {
  skip_on_cran()
  budget <- emdi_cores(3L)
  skip_if(budget < 2L, "machine reports a single core")
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")

  seen <- integer(0)
  orig <- monte_carlo_cpp            # capture BEFORE mocking, or this recurses
  testthat::with_mocked_bindings(
    {
      invisible(suppressMessages(ebp(
        fixed = eqIncome ~ gender + eqsize,
        pop_data = eusilcA_pop, pop_domains = "district",
        smp_data = eusilcA_smp, smp_domains = "district",
        threshold = 10924.32, transformation = "log",
        L = 5, MSE = FALSE, cpus = 3)))
    },
    monte_carlo_cpp = function(..., threads = 1L) {
      seen <<- c(seen, as.integer(threads))
      orig(..., threads = threads)
    },
    .package = "emdi"
  )
  expect_identical(seen, budget)
})

test_that("ebp() hands the resolved budget to the bootstrap kernel", {
  skip_on_cran()
  budget <- emdi_cores(2L)
  skip_if(budget < 2L, "machine reports a single core")
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")

  seen <- integer(0)
  orig <- parametric_bootstrap_cpp   # capture BEFORE mocking, or this recurses
  testthat::with_mocked_bindings(
    {
      set.seed(1)
      invisible(suppressMessages(ebp(
        fixed = eqIncome ~ gender + eqsize,
        pop_data = eusilcA_pop, pop_domains = "district",
        smp_data = eusilcA_smp, smp_domains = "district",
        threshold = 10924.32, transformation = "log",
        L = 5, MSE = TRUE, B = 3, cpus = 2)))
    },
    parametric_bootstrap_cpp = function(..., threads = 1L) {
      seen <<- c(seen, as.integer(threads))
      orig(..., threads = threads)
    },
    .package = "emdi"
  )
  expect_identical(seen, budget)
})

# The other half of the "never both" rule. The wild bootstrap has no C++ fast
# path, so the budget there is spent on worker processes; every bootstrap
# iteration must then run its point estimation single-threaded no matter how
# large a budget parametric_bootstrap() was handed. This is a guard rather
# than a driver -- it holds today because point_estim() defaults threads to 1 --
# and it fails the moment someone plumbs the budget through mse_estim() too.
test_that("the R bootstrap fallback pins its point estimation to one thread", {
  skip_on_cran()
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")
  f <- eqIncome ~ gender + eqsize
  framework <- framework_ebp(
    fixed = f, pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, custom_indicator = NULL,
    na.rm = TRUE, pop_weights = NULL, weights = NULL
  )
  set.seed(11)
  pe <- point_estim(framework = framework, fixed = f, transformation = "log",
                    interval = "default", L = 5)

  seen <- integer(0)
  orig <- monte_carlo_cpp            # capture BEFORE mocking, or this recurses
  testthat::with_mocked_bindings(
    {
      set.seed(11)
      invisible(suppressMessages(parametric_bootstrap(
        framework = framework, point_estim = pe, fixed = f,
        transformation = "log", interval = "default", L = 5, B = 2,
        boot_type = "wild", parallel_mode = "multicore", cpus = 1,
        control = NULL, true_indicators = NULL, threads = 4L)))
    },
    monte_carlo_cpp = function(..., threads = 1L) {
      seen <<- c(seen, as.integer(threads))
      orig(..., threads = threads)
    },
    .package = "emdi"
  )
  expect_length(seen, 2L)            # one per bootstrap iteration, not vacuous
  expect_true(all(seen == 1L))
})

# ebp() has to predict, before framework_ebp() has run, exactly what
# parametric_bootstrap() will later decide -- otherwise it could skip the
# L'Ecuyer switch while worker processes were in fact forked, leaving them all
# drawing from one shared RNG stream. This pins the two counts together.
test_that("ebp() counts indicators the same way framework_ebp() does", {
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")
  f <- eqIncome ~ gender + eqsize
  my_max <- function(y, pop_weights, threshold) max(y)

  for (ci in list(NULL, list(), list(my_max = my_max),
                  list(a = my_max, b = my_max), list(my_max))) {
    fw <- suppressMessages(framework_ebp(
      fixed = f, pop_data = eusilcA_pop, pop_domains = "district",
      smp_data = eusilcA_smp, smp_domains = "district",
      threshold = 10924.32, custom_indicator = ci,
      na.rm = TRUE, pop_weights = NULL, weights = NULL
    ))
    expect_identical(length(fw$indicator_names),
                     10L + length(names(ci)))
  }
})

# emdi_cores()'s documented precedence is argument > emdi.cores option >
# OMP_NUM_THREADS > 1. The env-var tier is only reachable from ebp() if ebp()
# leaves cpus alone and lets emdi_cores() do the whole resolution: a default
# that resolved the option itself (getOption("emdi.cores", 1L)) would hand
# emdi_cores() a non-NULL 1, so the tier below it could never be consulted and
# pipelines that set OMP_NUM_THREADS would silently drop to one core.
test_that("ebp() reaches the OMP_NUM_THREADS tier when cpus is left at its default", {
  skip_on_cran()
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")

  # The default must stay unresolved for the tiers below it to stay live.
  expect_null(formals(ebp)$cpus)

  withr::with_options(list(emdi.cores = NULL), {
    withr::with_envvar(c(OMP_NUM_THREADS = "2", `_R_CHECK_LIMIT_CORES_` = NA), {
      budget <- emdi_cores()
      skip_if(budget < 2L, "machine reports a single core")
      expect_identical(budget, 2L)

      seen <- integer(0)
      orig <- monte_carlo_cpp          # capture BEFORE mocking, or this recurses
      testthat::with_mocked_bindings(
        {
          invisible(suppressMessages(ebp(
            fixed = eqIncome ~ gender + eqsize,
            pop_data = eusilcA_pop, pop_domains = "district",
            smp_data = eusilcA_smp, smp_domains = "district",
            threshold = 10924.32, transformation = "log",
            L = 5, MSE = FALSE)))      # cpus deliberately not supplied
        },
        monte_carlo_cpp = function(..., threads = 1L) {
          seen <<- c(seen, as.integer(threads))
          orig(..., threads = threads)
        },
        .package = "emdi"
      )
      expect_identical(seen, 2L)
    })
  })
})

# ---------------------------------------------------------------------------
# fh() cpus core-budget argument (Task 7). fh() has no parallelMap branch, so
# its whole budget maps to OpenMP threads inside the C++ kernels.
# ---------------------------------------------------------------------------

# Same reasoning as the ebp() version above: fh()'s cpus default must be NULL,
# not a default that already calls getOption(...) itself, or the
# OMP_NUM_THREADS tier below it in emdi_cores()'s precedence would be
# unreachable.
test_that("fh() reaches the OMP_NUM_THREADS tier when cpus is left at its default", {
  skip_on_cran()
  data("eusilcA_popAgg", package = "emdi")
  data("eusilcA_smpAgg", package = "emdi")
  combined <- combine_data(eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain")

  expect_null(formals(fh)$cpus)

  withr::with_options(list(emdi.cores = NULL), {
    withr::with_envvar(c(OMP_NUM_THREADS = "2", `_R_CHECK_LIMIT_CORES_` = NA), {
      budget <- emdi_cores()
      skip_if(budget < 2L, "machine reports a single core")
      expect_identical(budget, 2L)

      seen <- integer(0)
      orig <- fh_jackknife_cpp        # capture BEFORE mocking, or this recurses
      testthat::with_mocked_bindings(
        {
          invisible(suppressMessages(fh(
            MTMED ~ cash + self_empl, vardir = "Var_MTMED",
            combined_data = combined, domains = "Domain", method = "reml",
            interval = c(0, 1e7), transformation = "arcsin",
            backtransformation = "naive", eff_smpsize = "n", MSE = TRUE,
            mse_type = "jackknife"
          )))      # cpus deliberately not supplied
        },
        fh_jackknife_cpp = function(..., threads = 1L) {
          seen <<- c(seen, as.integer(threads))
          orig(..., threads = threads)
        },
        .package = "emdi"
      )
      expect_identical(seen, 2L)
    })
  })
})

# is.numeric(NULL) is FALSE, so a cpus check written as
# `!is.numeric(cpus) -> stop(...)` without a NULL exemption would reject the
# documented default and error on every default call. This pins both halves:
# the typo is still caught, and NULL still passes through untouched.
test_that("fh() rejects a non-numeric cpus but accepts the NULL default", {
  data("eusilcA_popAgg", package = "emdi")
  data("eusilcA_smpAgg", package = "emdi")
  combined <- combine_data(eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain")
  fixed <- MTMED ~ cash + self_empl

  expect_error(
    suppressMessages(fh(
      fixed, vardir = "Var_MTMED", combined_data = combined,
      domains = "Domain", method = "reml", cpus = "four"
    )),
    regexp = "[Cc]pus"
  )

  expect_no_error(suppressMessages(fh(
    fixed, vardir = "Var_MTMED", combined_data = combined,
    domains = "Domain", method = "reml", cpus = NULL
  )))
})
