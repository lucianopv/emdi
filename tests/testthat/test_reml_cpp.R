test_that("std_transform_y_cpp matches R box_cox_std", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))
  for (lam in c(0.2, 0.5, 0.7, 1.0, 1.5, -0.5)) {
    r_result <- box_cox_std(y, lam)
    cpp_result <- std_transform_y_cpp(y, "box.cox", lam)
    expect_equal(as.numeric(cpp_result), as.numeric(r_result),
                 tolerance = 1e-10, info = paste("box.cox lambda =", lam))
  }
  # lambda ~ 0
  r_result_0 <- box_cox_std(y, 0)
  cpp_result_0 <- std_transform_y_cpp(y, "box.cox", 0)
  expect_equal(as.numeric(cpp_result_0), as.numeric(r_result_0),
               tolerance = 1e-10, info = "box.cox lambda = 0")
})

test_that("std_transform_y_cpp matches R dual_std", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))
  for (lam in c(0.2, 0.5, 0.7, 1.0, 1.5)) {
    r_result <- dual_std(y, lam)
    cpp_result <- std_transform_y_cpp(y, "dual", lam)
    expect_equal(as.numeric(cpp_result), as.numeric(r_result),
                 tolerance = 1e-10, info = paste("dual lambda =", lam))
  }
  r_result_0 <- dual_std(y, 0)
  cpp_result_0 <- std_transform_y_cpp(y, "dual", 0)
  expect_equal(as.numeric(cpp_result_0), as.numeric(r_result_0),
               tolerance = 1e-10, info = "dual lambda = 0")
})

test_that("std_transform_y_cpp matches R log_shift_opt_std", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))
  for (lam in c(100, 500, 1000, 5000)) {
    r_result <- log_shift_opt_std(y, lam)
    cpp_result <- std_transform_y_cpp(y, "log.shift", lam)
    expect_equal(as.numeric(cpp_result), as.numeric(r_result),
                 tolerance = 1e-10, info = paste("log.shift lambda =", lam))
  }
})

test_that("std_transform_y_cpp handles negative values with shift", {
  y <- c(-5, -2, 0, 3, 10, 50)
  r_result <- box_cox_std(y, 0.5)
  cpp_result <- std_transform_y_cpp(y, "box.cox", 0.5)
  expect_equal(as.numeric(cpp_result), as.numeric(r_result), tolerance = 1e-10)
})

test_that("reml_loglik_cpp matches lme() REML log-likelihood", {
  data("eusilcA_smp", package = "emdi2")
  fixed <- eqIncome ~ gender + eqsize

  for (lam in c(0.3, 0.5, 0.7, 1.0)) {
    sd_data <- std_data_transformation(
      fixed = fixed, smp_data = eusilcA_smp,
      transformation = "box.cox", lambda = lam
    )
    model <- nlme::lme(
      fixed = fixed, data = sd_data,
      random = ~ 1 | as.factor(district), method = "REML",
      keep.data = FALSE, control = nlme::lmeControl(opt = "optim")
    )
    lme_nll <- -as.numeric(logLik(model))

    # Data must be sorted by domain for C++
    smp_sorted <- eusilcA_smp[order(eusilcA_smp$district), ]
    y <- as.numeric(smp_sorted$eqIncome)
    X <- model.matrix(fixed, smp_sorted)
    domain_ids <- as.integer(as.factor(smp_sorted$district))
    n_d <- as.integer(table(as.factor(smp_sorted$district)))

    cpp_nll <- reml_loglik_cpp(
      lambda = lam, y = y, X = X,
      domain_ids = domain_ids, n_d = n_d,
      transformation = "box.cox"
    )

    expect_equal(cpp_nll, lme_nll, tolerance = 1e-4,
                 info = paste("box.cox lambda =", lam))
  }
})

test_that("reml_loglik_cpp works with dual transformation", {
  data("eusilcA_smp", package = "emdi2")
  fixed <- eqIncome ~ gender + eqsize

  lam <- 0.5
  sd_data <- std_data_transformation(
    fixed = fixed, smp_data = eusilcA_smp,
    transformation = "dual", lambda = lam
  )
  model <- nlme::lme(
    fixed = fixed, data = sd_data,
    random = ~ 1 | as.factor(district), method = "REML",
    keep.data = FALSE, control = nlme::lmeControl(opt = "optim")
  )
  lme_nll <- -as.numeric(logLik(model))

  smp_sorted <- eusilcA_smp[order(eusilcA_smp$district), ]
  y <- as.numeric(smp_sorted$eqIncome)
  X <- model.matrix(fixed, smp_sorted)
  domain_ids <- as.integer(as.factor(smp_sorted$district))
  n_d <- as.integer(table(as.factor(smp_sorted$district)))

  cpp_nll <- reml_loglik_cpp(
    lambda = lam, y = y, X = X,
    domain_ids = domain_ids, n_d = n_d,
    transformation = "dual"
  )
  expect_equal(cpp_nll, lme_nll, tolerance = 1e-4)
})

test_that("optimal_parameter_cpp matches R optimal_parameter for box.cox", {
  data("eusilcA_smp", package = "emdi2")
  fixed <- eqIncome ~ gender + eqsize

  r_lambda <- optimal_parameter(
    generic_opt = generic_opt,
    fixed = fixed,
    smp_data = eusilcA_smp,
    smp_domains = "district",
    transformation = "box.cox",
    interval = "default",
    control = list()
  )

  smp_data_sorted <- eusilcA_smp[order(eusilcA_smp$district), ]
  y_sorted <- as.numeric(smp_data_sorted$eqIncome)
  X_sorted <- model.matrix(fixed, smp_data_sorted)
  domain_ids <- as.integer(as.factor(smp_data_sorted$district))
  n_d <- as.integer(table(as.factor(smp_data_sorted$district)))

  cpp_lambda <- optimal_parameter_cpp(
    y = y_sorted, X = X_sorted,
    domain_ids = domain_ids, n_d = n_d,
    transformation = "box.cox",
    lower = -1, upper = 2
  )

  expect_equal(cpp_lambda, r_lambda, tolerance = 1e-4)
})

test_that("optimal_parameter_cpp matches R optimal_parameter for dual", {
  data("eusilcA_smp", package = "emdi2")
  fixed <- eqIncome ~ gender + eqsize

  r_lambda <- optimal_parameter(
    generic_opt = generic_opt,
    fixed = fixed,
    smp_data = eusilcA_smp,
    smp_domains = "district",
    transformation = "dual",
    interval = "default",
    control = list()
  )

  smp_data_sorted <- eusilcA_smp[order(eusilcA_smp$district), ]
  y_sorted <- as.numeric(smp_data_sorted$eqIncome)
  X_sorted <- model.matrix(fixed, smp_data_sorted)
  domain_ids <- as.integer(as.factor(smp_data_sorted$district))
  n_d <- as.integer(table(as.factor(smp_data_sorted$district)))

  cpp_lambda <- optimal_parameter_cpp(
    y = y_sorted, X = X_sorted,
    domain_ids = domain_ids, n_d = n_d,
    transformation = "dual",
    lower = 0, upper = 2
  )

  expect_equal(cpp_lambda, r_lambda, tolerance = 1e-4)
})

test_that("optimal_parameter_cpp matches R for full-model formula", {
  data("eusilcA_smp", package = "emdi2")
  fixed <- eqIncome ~ gender + eqsize + cash + self_empl +
    unempl_ben + age_ben + surv_ben + sick_ben + dis_ben +
    rent + fam_allow + house_allow + cap_inv + tax_adj

  r_lambda <- optimal_parameter(
    generic_opt = generic_opt,
    fixed = fixed,
    smp_data = eusilcA_smp,
    smp_domains = "district",
    transformation = "box.cox",
    interval = "default",
    control = list()
  )

  smp_data_sorted <- eusilcA_smp[order(eusilcA_smp$district), ]
  y_sorted <- as.numeric(smp_data_sorted$eqIncome)
  X_sorted <- model.matrix(fixed, smp_data_sorted)
  domain_ids <- as.integer(as.factor(smp_data_sorted$district))
  n_d <- as.integer(table(as.factor(smp_data_sorted$district)))

  cpp_lambda <- optimal_parameter_cpp(
    y = y_sorted, X = X_sorted,
    domain_ids = domain_ids, n_d = n_d,
    transformation = "box.cox",
    lower = -1, upper = 2
  )

  expect_equal(cpp_lambda, r_lambda, tolerance = 1e-4)
})

test_that("Full ebp() with C++ REML produces valid results", {
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

  ind <- estimators(result, indicator = "all")
  expect_true(all(ind$ind$Mean > 0))
  expect_true(all(ind$ind$Head_Count >= 0 & ind$ind$Head_Count <= 1))
  expect_true(all(ind$ind$Gini >= 0 & ind$ind$Gini <= 1))
})

test_that("Full ebp() with MSE and C++ REML works", {
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

  mse <- estimators(result, indicator = "all", MSE = TRUE)
  mse_cols <- grep("_MSE$", names(mse$ind), value = TRUE)
  for (col in mse_cols) {
    expect_true(all(mse$ind[[col]] >= 0), info = paste("MSE column:", col))
  }
})
