# Tests for C++ implementations against R reference functions

test_that("back_transform_cpp matches R back_transformation for all types", {
  set.seed(42)
  y <- rnorm(1000, 5, 2)
  lambda <- 0.7
  shift <- 1.5

  # no transformation
  expect_equal(
    back_transform_cpp(y, "no", 0, 0),
    as.numeric(y)
  )

  # log transformation
  expect_equal(
    back_transform_cpp(y, "log", 0, shift),
    exp(y) - shift
  )

  # box.cox transformation (lambda != 0)
  expect_equal(
    back_transform_cpp(y, "box.cox", lambda, shift),
    (lambda * y + 1)^(1 / lambda) - shift
  )

  # box.cox transformation (lambda ~ 0)
  expect_equal(
    back_transform_cpp(y, "box.cox", 0, shift),
    exp(y) - shift
  )

  # dual transformation (lambda != 0)
  expect_equal(
    back_transform_cpp(y, "dual", lambda, shift),
    (lambda * y + sqrt(lambda^2 * y^2 + 1))^(1 / lambda) - shift
  )

  # dual transformation (lambda ~ 0)
  expect_equal(
    back_transform_cpp(y, "dual", 0, shift),
    exp(y) - shift
  )

  # log.shift transformation
  expect_equal(
    back_transform_cpp(y, "log.shift", lambda, 0),
    exp(y) - lambda
  )
})

test_that("compute_domain_indicators_cpp matches R indicators (unweighted)", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))
  w <- rep(1, length(y))
  threshold <- 10000

  result <- compute_domain_indicators_cpp(y, w, threshold)

  # Mean
  expect_equal(result[1], mean(y), tolerance = 1e-10)
  # HCR
  expect_equal(result[2], mean(y < threshold), tolerance = 1e-10)
  # Poverty Gap
  pgap_r <- sum((1 - (y[y < threshold] / threshold)) * w[y < threshold]) / sum(w)
  expect_equal(result[3], pgap_r, tolerance = 1e-10)
  # Quantiles (unweighted uses R's quantile type=7)
  q_r <- as.numeric(quantile(y, probs = c(0.10, 0.25, 0.50, 0.75, 0.90)))
  expect_equal(result[6:10], q_r, tolerance = 1e-10)
})

test_that("compute_domain_indicators_cpp matches R indicators (weighted)", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))
  w <- runif(500, 0.5, 3.0)
  threshold <- 10000

  result <- compute_domain_indicators_cpp(y, w, threshold)

  # Weighted mean
  expect_equal(result[1], weighted.mean(y, w), tolerance = 1e-10)
  # Weighted HCR
  expect_equal(result[2], weighted.mean(y < threshold, w), tolerance = 1e-10)
})

test_that("compute_all_indicators_cpp computes across domains correctly", {
  set.seed(42)
  n <- 1000
  y <- abs(rnorm(n, 20000, 8000))
  w <- rep(1, n)
  domain_ids <- rep(1:5, each = 200)
  threshold <- 10000

  result <- compute_all_indicators_cpp(y, w, as.integer(domain_ids), threshold, 5L)
  expect_equal(nrow(result), 5)
  expect_equal(ncol(result), 10)

  # Verify domain 1 matches single-domain computation
  idx1 <- domain_ids == 1
  single <- compute_domain_indicators_cpp(y[idx1], w[idx1], threshold)
  expect_equal(as.numeric(result[1, ]), as.numeric(single), tolerance = 1e-10)
})
