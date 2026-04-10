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

test_that("monte_carlo_cpp produces correct dimensions and reasonable values", {
  set.seed(100)
  N_pop <- 500
  N_dom_pop <- 5
  n_per_dom <- 100
  N_dom_smp <- 4  # 4 in-sample, 1 out-of-sample
  N_dom_unobs <- 1

  mu <- rnorm(N_pop, 10, 1)
  sigmae2 <- 0.5
  sigmau2 <- 0.3
  sigmav2 <- rep(0.1, N_dom_smp)  # one per in-sample domain

  domain_ids <- rep(1:N_dom_pop, each = n_per_dom)
  # Domain 5 is out-of-sample
  dist_obs_dom <- c(1L, 1L, 1L, 1L, 0L)
  obs_dom <- rep(dist_obs_dom, each = n_per_dom)
  n_pop_vec <- rep(as.integer(n_per_dom), N_dom_pop)
  pop_weights <- rep(1.0, N_pop)

  result <- monte_carlo_cpp(
    mu = mu, sigmae2 = sigmae2, sigmau2 = sigmau2,
    sigmav2 = sigmav2,
    domain_ids = as.integer(domain_ids),
    obs_dom = as.integer(obs_dom),
    dist_obs_dom = as.integer(dist_obs_dom),
    n_pop = n_pop_vec,
    N_dom_pop = N_dom_pop,
    N_dom_smp = N_dom_smp,
    N_dom_unobs = N_dom_unobs,
    L = 3L,
    threshold = 10000,
    transformation = "log",
    lambda = 0, shift = 0,
    pop_weights = pop_weights,
    n_indicators = 10L
  )

  # Check dimensions
  expect_equal(nrow(result$point_estimates), N_dom_pop)
  expect_equal(ncol(result$point_estimates), 10)
  expect_equal(nrow(result$y_mcmc), N_pop)
  expect_equal(ncol(result$y_mcmc), 3)

  # Means should be positive (exp of normal ~ lognormal)
  expect_true(all(result$point_estimates[, 1] > 0))
  # HCR should be between 0 and 1
  expect_true(all(result$point_estimates[, 2] >= 0))
  expect_true(all(result$point_estimates[, 2] <= 1))
})

test_that("gen_superpop_cpp produces correct dimensions", {
  set.seed(42)
  N_pop <- 500
  N_dom_pop <- 5
  n_per_dom <- 100
  mu_fixed <- rnorm(N_pop, 10, 1)
  obs_dom <- rep(c(1L, 1L, 1L, 1L, 0L), each = n_per_dom)
  n_pop_vec <- rep(as.integer(n_per_dom), N_dom_pop)

  result <- gen_superpop_cpp(
    mu_fixed = mu_fixed,
    sigmae2 = 0.5, sigmau2 = 0.3,
    obs_dom = obs_dom,
    n_pop = n_pop_vec,
    N_dom_pop = N_dom_pop,
    transformation = "log",
    lambda = 0, shift = 0
  )

  expect_equal(length(result$pop_income_vector), N_pop)
  expect_equal(length(result$vu_tmp), N_dom_pop)
  expect_equal(length(result$eps), N_pop)
  expect_equal(length(result$vu_pop), N_pop)
  expect_true(all(result$pop_income_vector >= 0))
})

test_that("gen_bootstrap_sample_cpp produces correct dimensions", {
  set.seed(99)
  N_smp <- 200
  N_dom_smp <- 4
  n_smp_vec <- rep(50L, N_dom_smp)
  p <- 3
  X_smp <- matrix(rnorm(N_smp * p), nrow = N_smp, ncol = p)
  betas <- rnorm(p)
  vu_tmp <- rnorm(5)  # 5 pop domains
  smp_to_pop_map <- as.integer(c(1, 2, 3, 4))  # all map to pop domains

  result <- gen_bootstrap_sample_cpp(
    X_smp = X_smp, betas = betas,
    sigmae2 = 0.5, sigmau2 = 0.3,
    vu_tmp = vu_tmp,
    smp_to_pop_map = smp_to_pop_map,
    n_smp = n_smp_vec,
    transformation = "log",
    lambda = 0, shift = 0
  )

  expect_equal(length(result), N_smp)
  expect_true(all(is.finite(result)))
  expect_true(all(result >= 0))
})

test_that("Full ebp() with C++ produces valid results", {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  set.seed(42)
  result <- ebp(
    fixed = eqIncome ~ gender + eqsize + cash + self_empl +
      unempl_ben + age_ben + surv_ben + sick_ben + dis_ben +
      rent + fam_allow + house_allow + cap_inv + tax_adj,
    pop_data = eusilcA_pop,
    pop_domains = "district",
    smp_data = eusilcA_smp,
    smp_domains = "district",
    L = 10,
    MSE = FALSE
  )

  # Basic sanity checks
  ind_obj <- estimators(result, indicator = "all")
  ind <- ind_obj$ind
  expect_true(all(ind$Mean > 0))
  expect_true(all(ind$Head_Count >= 0 & ind$Head_Count <= 1))
  expect_true(all(ind$Gini >= 0 & ind$Gini <= 1))
  expect_true(all(ind$Poverty_Gap >= 0))
})

test_that("Full ebp() with MSE and C++ produces valid results", {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  set.seed(42)
  result <- ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop,
    pop_domains = "district",
    smp_data = eusilcA_smp,
    smp_domains = "district",
    L = 5,
    MSE = TRUE,
    B = 3
  )

  # MSE should be non-negative
  mse_obj <- estimators(result, indicator = "all", MSE = TRUE)
  mse_df <- mse_obj$ind
  mse_cols <- grep("_MSE$", names(mse_df), value = TRUE)
  expect_true(length(mse_cols) > 0, info = "No MSE columns found")
  for (col in mse_cols) {
    expect_true(all(mse_df[[col]] >= 0), info = paste("MSE column:", col))
  }
})
