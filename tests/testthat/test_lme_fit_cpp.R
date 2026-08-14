test_that("data_transform_cpp matches R data_transformation for all types", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))

  # log
  r_log <- log_transform(y, shift = 0)
  cpp_log <- data_transform_cpp(y, "log", 0)
  expect_equal(as.numeric(cpp_log$y), as.numeric(r_log$y), tolerance = 1e-10)
  expect_equal(cpp_log$shift, r_log$shift, tolerance = 1e-10)

  # box.cox
  for (lam in c(0.3, 0.7, 1.0)) {
    r_bc <- box_cox(y, lambda = lam, shift = 0)
    cpp_bc <- data_transform_cpp(y, "box.cox", lam)
    expect_equal(as.numeric(cpp_bc$y), as.numeric(r_bc$y), tolerance = 1e-10,
                 info = paste("box.cox lambda =", lam))
    expect_equal(cpp_bc$shift, r_bc$shift, tolerance = 1e-10)
  }

  # box.cox lambda ~ 0
  r_bc0 <- box_cox(y, lambda = 0, shift = 0)
  cpp_bc0 <- data_transform_cpp(y, "box.cox", 0)
  expect_equal(as.numeric(cpp_bc0$y), as.numeric(r_bc0$y), tolerance = 1e-10)

  # dual
  r_dual <- dual(y, lambda = 0.5, shift = 0)
  cpp_dual <- data_transform_cpp(y, "dual", 0.5)
  expect_equal(as.numeric(cpp_dual$y), as.numeric(r_dual$y), tolerance = 1e-10)

  # no
  cpp_no <- data_transform_cpp(y, "no", 0)
  expect_equal(as.numeric(cpp_no$y), y, tolerance = 1e-10)
})

test_that("data_transform_cpp handles negative values with shift", {
  y <- c(-5, -2, 0, 3, 10, 50)
  r_bc <- box_cox(y, lambda = 0.5, shift = 0)
  cpp_bc <- data_transform_cpp(y, "box.cox", 0.5)
  expect_equal(as.numeric(cpp_bc$y), as.numeric(r_bc$y), tolerance = 1e-10)
  expect_equal(cpp_bc$shift, r_bc$shift, tolerance = 1e-10)
})

test_that("lme_fit_cpp matches nlme::lme() for unweighted case", {
  data("eusilcA_smp", package = "emdi")
  fixed <- eqIncome ~ gender + eqsize

  transformation_par <- data_transformation(
    fixed = fixed, smp_data = eusilcA_smp,
    transformation = "box.cox", lambda = 0.7
  )

  model <- nlme::lme(
    fixed = fixed, data = transformation_par$transformed_data,
    random = ~ 1 | as.factor(district), method = "REML",
    keep.data = FALSE, control = list()
  )

  lme_betas <- nlme::fixed.effects(model)
  lme_sigma2e <- model$sigma^2
  lme_sigma2u <- as.numeric(nlme::VarCorr(model)[1, 1])
  lme_re <- nlme::random.effects(model)[[1]]

  # C++ fit (data must be sorted by domain)
  smp_sorted <- eusilcA_smp[order(eusilcA_smp$district), ]
  tp <- data_transformation(fixed = fixed, smp_data = smp_sorted,
    transformation = "box.cox", lambda = 0.7)
  y_trans <- as.numeric(tp$transformed_data$eqIncome)
  X <- model.matrix(fixed, smp_sorted)
  domain_factor <- as.factor(smp_sorted$district)
  n_d <- as.integer(table(domain_factor))

  cpp_fit <- lme_fit_cpp(y_trans, X, n_d)

  expect_equal(as.numeric(cpp_fit$betas), as.numeric(lme_betas), tolerance = 1e-6)
  expect_equal(cpp_fit$sigma2_e, lme_sigma2e, tolerance = 1e-4)
  expect_equal(cpp_fit$sigma2_u, lme_sigma2u, tolerance = 1e-4)

  # Compare BLUPs - random.effects returns a data.frame with rownames = domain levels
  lme_re_df <- nlme::random.effects(model)
  smp_domain_names <- levels(domain_factor)
  lme_re_ordered <- lme_re_df[smp_domain_names, 1]
  expect_equal(as.numeric(cpp_fit$rand_eff), as.numeric(lme_re_ordered), tolerance = 1e-4)
})

test_that("lme_fit_cpp works with log transformation", {
  data("eusilcA_smp", package = "emdi")
  fixed <- eqIncome ~ gender + eqsize

  smp_sorted <- eusilcA_smp[order(eusilcA_smp$district), ]
  tp <- data_transformation(fixed = fixed, smp_data = smp_sorted,
    transformation = "log", lambda = NULL)
  y_trans <- as.numeric(tp$transformed_data$eqIncome)
  X <- model.matrix(fixed, smp_sorted)
  n_d <- as.integer(table(as.factor(smp_sorted$district)))

  model <- nlme::lme(fixed = fixed, data = tp$transformed_data,
    random = ~ 1 | as.factor(district), method = "REML", keep.data = FALSE)

  cpp_fit <- lme_fit_cpp(y_trans, X, n_d)

  expect_equal(cpp_fit$sigma2_e, model$sigma^2, tolerance = 1e-4)
  expect_equal(cpp_fit$sigma2_u, as.numeric(nlme::VarCorr(model)[1, 1]), tolerance = 1e-4)
})

test_that("lme_fit_cpp works with full model formula", {
  data("eusilcA_smp", package = "emdi")
  fixed <- eqIncome ~ gender + eqsize + cash + self_empl +
    unempl_ben + age_ben + surv_ben + sick_ben + dis_ben +
    rent + fam_allow + house_allow + cap_inv + tax_adj

  smp_sorted <- eusilcA_smp[order(eusilcA_smp$district), ]
  tp <- data_transformation(fixed = fixed, smp_data = smp_sorted,
    transformation = "box.cox", lambda = 0.6)
  y_trans <- as.numeric(tp$transformed_data$eqIncome)
  X <- model.matrix(fixed, smp_sorted)
  n_d <- as.integer(table(as.factor(smp_sorted$district)))

  model <- nlme::lme(fixed = fixed, data = tp$transformed_data,
    random = ~ 1 | as.factor(district), method = "REML", keep.data = FALSE)

  cpp_fit <- lme_fit_cpp(y_trans, X, n_d)

  expect_equal(as.numeric(cpp_fit$betas), as.numeric(nlme::fixed.effects(model)),
               tolerance = 1e-5)
  expect_equal(cpp_fit$sigma2_e, model$sigma^2, tolerance = 1e-3)
  expect_equal(cpp_fit$sigma2_u, as.numeric(nlme::VarCorr(model)[1, 1]), tolerance = 1e-3)
})

test_that("model_par_weighted_cpp matches R model_par weighted case", {
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")

  # Create synthetic weights
  set.seed(123)
  eusilcA_smp$weight <- runif(nrow(eusilcA_smp), 1.0, 3.0)
  fixed <- eqIncome ~ gender + eqsize

  framework <- framework_ebp(
    fixed = fixed, pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, custom_indicator = NULL,
    na.rm = TRUE, pop_weights = NULL, weights = "weight"
  )

  tp <- data_transformation(fixed = fixed, smp_data = framework$smp_data,
    transformation = "log", lambda = NULL)

  model <- nlme::lme(fixed = fixed, data = tp$transformed_data,
    random = ~ 1 | as.factor(district), method = "REML",
    keep.data = FALSE)

  r_par <- model_par(mixed_model = model, framework = framework,
    fixed = fixed, transformation_par = tp)

  # C++ version: sort data by domain
  smp_sorted <- framework$smp_data[order(framework$smp_data$district), ]
  tp_sorted <- data_transformation(fixed = fixed, smp_data = smp_sorted,
    transformation = "log", lambda = NULL)
  y_trans <- as.numeric(tp_sorted$transformed_data$eqIncome)
  X <- model.matrix(fixed, smp_sorted)
  n_d_smp <- as.integer(table(as.factor(smp_sorted$district)))
  w <- as.numeric(tp_sorted$transformed_data$weight)

  # Get sigma2_e, sigma2_u from C++ fit
  fit <- lme_fit_cpp(y_trans, X, n_d_smp)

  # Compute weighted parameters
  cpp_par <- model_par_weighted_cpp(y_trans, X, w, n_d_smp,
    fit$sigma2_e, fit$sigma2_u)

  # Compare betas
  expect_equal(as.numeric(cpp_par$betas), as.numeric(r_par$betas), tolerance = 1e-4)
  # Compare gamma_weight
  expect_equal(as.numeric(cpp_par$gammaw), as.numeric(r_par$gammaw), tolerance = 1e-4)
  # Compare delta2
  expect_equal(as.numeric(cpp_par$delta2), as.numeric(r_par$delta2), tolerance = 1e-6)
})
