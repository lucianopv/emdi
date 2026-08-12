test_that("emdi_cores defaults to 1 with no configuration", {
  withr::with_options(list(emdi2.cores = NULL), {
    withr::with_envvar(c(OMP_NUM_THREADS = NA, `_R_CHECK_LIMIT_CORES_` = NA), {
      expect_identical(emdi_cores(), 1L)
    })
  })
})

test_that("emdi_cores precedence is argument > option > env var", {
  withr::with_envvar(c(OMP_NUM_THREADS = "3", `_R_CHECK_LIMIT_CORES_` = NA), {
    withr::with_options(list(emdi2.cores = NULL), {
      expect_identical(emdi_cores(), 3L)              # env var used
    })
    withr::with_options(list(emdi2.cores = 2L), {
      expect_identical(emdi_cores(), 2L)              # option beats env
      expect_identical(emdi_cores(1L), 1L)            # argument beats option
    })
  })
})

test_that("emdi_cores caps at 2 under R CMD check", {
  withr::with_envvar(c(`_R_CHECK_LIMIT_CORES_` = "TRUE", OMP_NUM_THREADS = NA), {
    withr::with_options(list(emdi2.cores = NULL), {
      expect_identical(emdi_cores(64L), 2L)              # cap via argument
    })
    withr::with_options(list(emdi2.cores = 64L), {
      expect_identical(emdi_cores(), 2L)                # cap via option
    })
  })
})

test_that("emdi_cores never exceeds detectCores and never returns < 1", {
  withr::with_envvar(c(`_R_CHECK_LIMIT_CORES_` = NA, OMP_NUM_THREADS = NA), {
    withr::with_options(list(emdi2.cores = NULL), {
      expect_identical(emdi_cores(1000L), as.integer(parallel::detectCores()))
      expect_identical(emdi_cores(0L), 1L)
      expect_identical(emdi_cores(-5L), 1L)
    })
  })
})

test_that("emdi_cores ignores an unparseable OMP_NUM_THREADS", {
  withr::with_envvar(c(OMP_NUM_THREADS = "not-a-number",
                       `_R_CHECK_LIMIT_CORES_` = NA), {
    withr::with_options(list(emdi2.cores = NULL), {
      expect_identical(emdi_cores(), 1L)
    })
  })
})
