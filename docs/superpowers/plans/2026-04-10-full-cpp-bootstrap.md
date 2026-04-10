# Full C++ Parametric Bootstrap Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the entire parametric bootstrap B-iteration loop into a single C++ function, eliminating all per-iteration R overhead (~39% of iteration time).

**Architecture:** `parametric_bootstrap_cpp()` orchestrates the full loop in C++, calling existing C++ functions (MC simulation, indicators, superpopulation, bootstrap sample, REML optimization) plus new helper functions (forward transform, LME fit with BLUPs, gen_model computation, weighted model_par). R's `parametric_bootstrap()` dispatches to C++ when conditions are met (parametric boot, no custom indicators), otherwise falls back to existing R loop.

**Tech Stack:** R, Rcpp, RcppArmadillo (reuses existing Brent optimizer and REML machinery)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/lme_fit.cpp` | Create | Forward transforms, LME fit returning full params, weighted model_par, gen_model computation |
| `src/parametric_bootstrap.cpp` | Create | The full B-iteration bootstrap loop |
| `R/mse_estimation.R` | Modify | Dispatch to C++ in parametric_bootstrap() |
| `tests/testthat/test_lme_fit_cpp.R` | Create | Tests for LME fit and helpers |
| `tests/testthat/test_bootstrap_cpp.R` | Create | Tests for full bootstrap loop |

---

### Task 1: Forward Data Transformation in C++

**Files:**
- Create: `src/lme_fit.cpp`
- Create: `tests/testthat/test_lme_fit_cpp.R`

Implement the non-standardized forward transformation (with shift computation) that `data_transformation()` does. This is needed to transform the bootstrap sample before fitting the model.

The R functions to match:
- `"no"`: returns y unchanged, shift = NULL
- `"log"`: if min(y) <= 0, shift = abs(min) + 1, y = y + shift; y = log(y); shift defaults to 0
- `"box.cox"`: if min(y) <= 0, shift = abs(min) + 1, y = y + shift; y = ((y+shift)^lambda - 1)/lambda (or log(y+shift) if lambda~0)
- `"dual"`: if min(y) <= 0, shift = abs(min) + 1, y = y + shift; y = ((y+shift)^lambda - (y+shift)^(-lambda))/(2*lambda)
- `"log.shift"`: adjust lambda if min(y+lambda) <= 0; y = log(y + lambda); shift = NULL

- [ ] **Step 1: Write failing tests**

Create `tests/testthat/test_lme_fit_cpp.R`:
```r
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/home/lpv/Dropbox/Inglaterra_2019/PhD/Southampton/R Packages/emdi"
Rscript -e 'devtools::test(filter = "lme_fit_cpp")'
```

Expected: FAIL with "could not find function data_transform_cpp"

- [ ] **Step 3: Implement data_transform_cpp**

Create `src/lme_fit.cpp`:
```cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// ---------------------------------------------------------------------------
// Forward data transformation (non-standardized, with shift).
// Matches R data_transformation() -> box_cox() / dual() / log_transform() etc.
// Returns list(y = transformed_y, shift = shift_value)
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List data_transform_cpp(const arma::vec& y_raw,
                               const std::string& transformation,
                               double lambda) {
  int n = y_raw.n_elem;
  arma::vec y = y_raw;
  double shift = 0.0;
  bool shift_is_null = false;

  if (transformation == "no") {
    shift_is_null = true;
    // y unchanged

  } else if (transformation == "log") {
    double mn = y.min();
    if (mn <= 0) {
      shift = std::abs(mn) + 1.0;
      y = y + shift;
    }
    y = arma::log(y);

  } else if (transformation == "box.cox") {
    double mn = y.min();
    if (mn <= 0) {
      shift = std::abs(mn) + 1.0;
    }
    if (std::abs(lambda) <= 1e-12) {
      y = arma::log(y + shift);
    } else {
      y = (arma::pow(y + shift, lambda) - 1.0) / lambda;
    }

  } else if (transformation == "dual") {
    double mn = y.min();
    if (mn <= 0) {
      shift = std::abs(mn) + 1.0;
    }
    if (std::abs(lambda) <= 1e-12) {
      y = arma::log(y + shift);
    } else {
      arma::vec yps = y + shift;
      y = (arma::pow(yps, lambda) - arma::pow(yps, -lambda)) / (2.0 * lambda);
    }

  } else if (transformation == "log.shift") {
    // For log.shift, lambda IS the shift, and the returned shift is NULL
    double mn = arma::min(y + lambda);
    if (mn <= 0) {
      lambda = lambda + std::abs(y.min()) + 1.0;
    }
    y = arma::log(y + lambda);
    shift_is_null = true;

  } else {
    Rcpp::stop("Unknown transformation: " + transformation);
  }

  if (shift_is_null) {
    return Rcpp::List::create(
      Rcpp::Named("y") = y,
      Rcpp::Named("shift") = R_NilValue
    );
  } else {
    return Rcpp::List::create(
      Rcpp::Named("y") = y,
      Rcpp::Named("shift") = shift
    );
  }
}
```

- [ ] **Step 4: Compile and run tests**

```bash
Rscript -e 'Rcpp::compileAttributes(".")'
Rscript -e 'pkgbuild::compile_dll(".")'
Rscript -e 'devtools::test(filter = "lme_fit_cpp")'
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lme_fit.cpp tests/testthat/test_lme_fit_cpp.R R/RcppExports.R src/RcppExports.cpp
git commit -m "feat: add C++ forward data transformation with shift"
```

---

### Task 2: LME Fit Returning Full Parameters

**Files:**
- Modify: `src/lme_fit.cpp`
- Modify: `tests/testthat/test_lme_fit_cpp.R`

Create `lme_fit_cpp()` that fits the random-intercept REML model and returns betas, sigma2_e, sigma2_u, and BLUPs. This reuses the REML machinery from `reml_optimization.cpp` but extends it to return all model parameters.

- [ ] **Step 1: Write failing test**

Append to `tests/testthat/test_lme_fit_cpp.R`:
```r
test_that("lme_fit_cpp matches nlme::lme() for unweighted case", {
  data("eusilcA_smp", package = "emdi2")
  fixed <- eqIncome ~ gender + eqsize

  # Transform with box.cox
  transformation_par <- data_transformation(
    fixed = fixed, smp_data = eusilcA_smp,
    transformation = "box.cox", lambda = 0.7
  )

  # nlme fit
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

  expect_equal(as.numeric(cpp_fit$betas), as.numeric(lme_betas),
               tolerance = 1e-6)
  expect_equal(cpp_fit$sigma2_e, lme_sigma2e, tolerance = 1e-4)
  expect_equal(cpp_fit$sigma2_u, lme_sigma2u, tolerance = 1e-4)

  # Compare BLUPs (C++ returns one per domain in order, lme has names)
  smp_domain_names <- levels(domain_factor)
  lme_re_ordered <- lme_re[smp_domain_names]
  expect_equal(as.numeric(cpp_fit$rand_eff), as.numeric(lme_re_ordered),
               tolerance = 1e-4)
})

test_that("lme_fit_cpp works with log transformation", {
  data("eusilcA_smp", package = "emdi2")
  fixed <- eqIncome ~ gender + eqsize

  smp_sorted <- eusilcA_smp[order(eusilcA_smp$district), ]
  tp <- data_transformation(fixed = fixed, smp_data = smp_sorted,
    transformation = "log", lambda = NULL)
  y_trans <- as.numeric(tp$transformed_data$eqIncome)
  X <- model.matrix(fixed, smp_sorted)
  n_d <- as.integer(table(as.factor(smp_sorted$district)))

  model <- nlme::lme(fixed = fixed, data = tp$transformed_data,
    random = ~ 1 | as.factor(district), method = "REML",
    keep.data = FALSE)

  cpp_fit <- lme_fit_cpp(y_trans, X, n_d)

  expect_equal(cpp_fit$sigma2_e, model$sigma^2, tolerance = 1e-4)
  expect_equal(cpp_fit$sigma2_u,
               as.numeric(nlme::VarCorr(model)[1, 1]), tolerance = 1e-4)
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
Rscript -e 'devtools::test(filter = "lme_fit_cpp")'
```

- [ ] **Step 3: Implement lme_fit_cpp**

Add to `src/lme_fit.cpp`:
```cpp
// Forward declarations from reml_optimization.cpp
arma::vec std_transform_y_cpp(const arma::vec& y_raw,
                               const std::string& transformation,
                               double lambda);

// Brent's method (replicated here as static since templates can't be
// forward-declared across translation units)
template <typename F>
static double brent_minimize_local(F f, double a, double b,
                                    double tol = 1.490116e-08,
                                    int maxiter = 1000) {
  const double golden = 0.3819660112501051;
  double x = a + golden * (b - a);
  double w = x, v = x;
  double fx = f(x), fw = fx, fv = fx;
  double d = 0.0, e = 0.0;
  for (int iter = 0; iter < maxiter; ++iter) {
    double midpoint = 0.5 * (a + b);
    double tol1 = tol * std::abs(x) + 1e-10;
    double tol2 = 2.0 * tol1;
    if (std::abs(x - midpoint) <= (tol2 - 0.5 * (b - a))) return x;
    double p = 0.0, q = 0.0, r = 0.0;
    if (std::abs(e) > tol1) {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p; else q = -q;
      r = e; e = d;
    }
    if (std::abs(p) < std::abs(0.5 * q * r) &&
        p > q * (a - x) && p < q * (b - x)) {
      d = p / q;
      double u = x + d;
      if ((u - a) < tol2 || (b - u) < tol2)
        d = (x < midpoint) ? tol1 : -tol1;
    } else {
      e = (x < midpoint) ? b - x : a - x;
      d = golden * e;
    }
    double u = (std::abs(d) >= tol1) ? x + d : x + ((d > 0) ? tol1 : -tol1);
    double fu = f(u);
    if (fu <= fx) {
      if (u < x) b = x; else a = x;
      v = w; fv = fw; w = x; fw = fx; x = u; fx = fu;
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) { v = w; fv = fw; w = u; fw = fu; }
      else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
    }
  }
  return x;
}

// ---------------------------------------------------------------------------
// Per-domain sufficient statistics (shared by lme_fit and bootstrap)
// ---------------------------------------------------------------------------
struct DomainSuffStats {
  arma::mat S_xx;   // X_d' X_d
  arma::vec S_xy;   // X_d' y_d
  double S_yy;      // y_d' y_d
  arma::vec xbar;   // colSums(X_d)
  double ybar;      // sum(y_d)
  int nd;
};

static std::vector<DomainSuffStats> compute_domain_stats(
    const arma::vec& y, const arma::mat& X, const arma::ivec& n_d, int D) {
  std::vector<DomainSuffStats> stats(D);
  int offset = 0;
  for (int d = 0; d < D; d++) {
    int nd = n_d(d);
    stats[d].nd = nd;
    arma::mat X_d = X.rows(offset, offset + nd - 1);
    arma::vec y_d = y.subvec(offset, offset + nd - 1);
    stats[d].S_xx = X_d.t() * X_d;
    stats[d].S_xy = X_d.t() * y_d;
    stats[d].S_yy = arma::dot(y_d, y_d);
    stats[d].xbar = arma::sum(X_d, 0).t();
    stats[d].ybar = arma::accu(y_d);
    offset += nd;
  }
  return stats;
}

// Evaluate profile REML and return all parameters at given theta
struct RemlResult {
  arma::vec beta;
  double sigma2_e;
  double sigma2_u;
  double neg_reml_loglik;
};

static RemlResult reml_at_theta(double theta,
                                 const std::vector<DomainSuffStats>& stats,
                                 int n, int p, int D) {
  arma::mat A(p, p, arma::fill::zeros);
  arma::vec b_vec(p, arma::fill::zeros);
  double log_det_V_part = 0.0;

  for (int d = 0; d < D; d++) {
    double c_d = theta / (1.0 + stats[d].nd * theta);
    A += stats[d].S_xx - c_d * (stats[d].xbar * stats[d].xbar.t());
    b_vec += stats[d].S_xy - c_d * stats[d].xbar * stats[d].ybar;
    log_det_V_part += std::log(1.0 + stats[d].nd * theta);
  }

  RemlResult res;
  bool solved = arma::solve(res.beta, A, b_vec, arma::solve_opts::likely_sympd);
  if (!solved) {
    res.neg_reml_loglik = 1e30;
    res.sigma2_e = 1.0;
    res.sigma2_u = 0.0;
    return res;
  }

  double quad = 0.0;
  for (int d = 0; d < D; d++) {
    double c_d = theta / (1.0 + stats[d].nd * theta);
    double rss_d = stats[d].S_yy
      - 2.0 * arma::dot(res.beta, stats[d].S_xy)
      + arma::as_scalar(res.beta.t() * stats[d].S_xx * res.beta);
    double rbar_d = stats[d].ybar - arma::dot(stats[d].xbar, res.beta);
    quad += rss_d - c_d * rbar_d * rbar_d;
  }

  res.sigma2_e = quad / (double)(n - p);
  if (res.sigma2_e <= 0) res.sigma2_e = 1e-10;
  res.sigma2_u = theta * res.sigma2_e;

  double log_det_V = n * std::log(res.sigma2_e) + log_det_V_part;
  double log_det_A_val, log_det_A_sign;
  arma::log_det(log_det_A_val, log_det_A_sign, A);
  double log_det_XVX = log_det_A_val - p * std::log(res.sigma2_e);

  res.neg_reml_loglik = 0.5 * ((n - p) * std::log(2.0 * M_PI)
    + log_det_V + log_det_XVX + (double)(n - p));

  return res;
}

// ---------------------------------------------------------------------------
// lme_fit_cpp: Fit random-intercept REML model, return all parameters.
//
// y_transformed: already-transformed response [N_smp] (sorted by domain)
// X: model matrix [N_smp x p] (sorted by domain)
// n_d: sample count per domain [D]
//
// Returns list: betas, sigma2_e, sigma2_u, rand_eff (BLUPs), gamma
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List lme_fit_cpp(const arma::vec& y_transformed,
                        const arma::mat& X,
                        const arma::ivec& n_d) {
  int n = y_transformed.n_elem;
  int p = X.n_cols;
  int D = n_d.n_elem;

  // Compute sufficient statistics
  auto stats = compute_domain_stats(y_transformed, X, n_d, D);

  // Optimize over log(theta)
  auto neg_reml = [&](double log_theta) -> double {
    double theta = std::exp(log_theta);
    return reml_at_theta(theta, stats, n, p, D).neg_reml_loglik;
  };

  double best_log_theta = brent_minimize_local(neg_reml, -15.0, 15.0);
  double best_theta = std::exp(best_log_theta);

  // Get parameters at optimal theta
  RemlResult res = reml_at_theta(best_theta, stats, n, p, D);

  // Compute BLUPs: u_hat_d = gamma_d * (y_bar_d - x_bar_d' * beta)
  // gamma_d = sigma2_u / (sigma2_u + sigma2_e / n_d)
  arma::vec rand_eff(D);
  arma::vec gamma_vec(D);
  for (int d = 0; d < D; d++) {
    gamma_vec(d) = res.sigma2_u / (res.sigma2_u + res.sigma2_e / stats[d].nd);
    double y_bar_d = stats[d].ybar / stats[d].nd;
    arma::vec x_bar_d = stats[d].xbar / stats[d].nd;
    rand_eff(d) = gamma_vec(d) * (y_bar_d - arma::dot(x_bar_d, res.beta));
  }

  return Rcpp::List::create(
    Rcpp::Named("betas") = res.beta,
    Rcpp::Named("sigma2_e") = res.sigma2_e,
    Rcpp::Named("sigma2_u") = res.sigma2_u,
    Rcpp::Named("rand_eff") = rand_eff,
    Rcpp::Named("gamma") = gamma_vec
  );
}
```

- [ ] **Step 4: Compile and run tests**

```bash
Rscript -e 'Rcpp::compileAttributes(".")'
Rscript -e 'pkgbuild::compile_dll(".")'
Rscript -e 'devtools::test(filter = "lme_fit_cpp")'
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lme_fit.cpp tests/testthat/test_lme_fit_cpp.R R/RcppExports.R src/RcppExports.cpp
git commit -m "feat: add lme_fit_cpp - C++ random-intercept REML with BLUPs"
```

---

### Task 3: Weighted Model Parameters in C++

**Files:**
- Modify: `src/lme_fit.cpp`
- Modify: `tests/testthat/test_lme_fit_cpp.R`

The weighted pseudo-EB approach uses sigma2_e/sigma2_u from the model fit, then recomputes betas and random effects using survey weights.

- [ ] **Step 1: Write failing test**

Append to `tests/testthat/test_lme_fit_cpp.R`:
```r
test_that("model_par_weighted_cpp matches R model_par weighted case", {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  # Create framework with weights
  eusilcA_smp$weight <- runif(nrow(eusilcA_smp), 0.5, 2.0)
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

  # C++ version
  smp_sorted <- framework$smp_data[order(framework$smp_data$district), ]
  tp_sorted <- data_transformation(fixed = fixed, smp_data = smp_sorted,
    transformation = "log", lambda = NULL)
  y_trans <- as.numeric(tp_sorted$transformed_data$eqIncome)
  X <- model.matrix(fixed, smp_sorted)
  n_d <- as.integer(table(as.factor(smp_sorted$district)))
  w <- as.numeric(tp_sorted$transformed_data$weight)

  # First get sigma2_e, sigma2_u from C++ fit
  fit <- lme_fit_cpp(y_trans, X, n_d)

  # Then compute weighted parameters
  cpp_par <- model_par_weighted_cpp(y_trans, X, w, n_d,
    fit$sigma2_e, fit$sigma2_u)

  expect_equal(as.numeric(cpp_par$betas), as.numeric(r_par$betas),
               tolerance = 1e-4)
  expect_equal(as.numeric(cpp_par$gammaw), as.numeric(r_par$gammaw),
               tolerance = 1e-4)
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
Rscript -e 'devtools::test(filter = "lme_fit_cpp")'
```

- [ ] **Step 3: Implement model_par_weighted_cpp**

Add to `src/lme_fit.cpp`:
```cpp
// ---------------------------------------------------------------------------
// model_par_weighted_cpp: Compute weighted pseudo-EB parameters.
// Takes sigma2_e/sigma2_u from an LME fit and recomputes betas and
// random effects using survey weights.
// Matches R model_par() weighted path (point_estimation.R:172-266).
//
// y_transformed, X, weights must be sorted by domain.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List model_par_weighted_cpp(const arma::vec& y_transformed,
                                   const arma::mat& X,
                                   const arma::vec& weights,
                                   const arma::ivec& n_d,
                                   double sigma2_e,
                                   double sigma2_u) {
  int p = X.n_cols;
  int D = n_d.n_elem;

  arma::vec delta2(D);
  arma::vec gamma_weight(D);
  arma::mat num(p, 1, arma::fill::zeros);
  arma::mat den(p, p, arma::fill::zeros);
  arma::vec mean_dep(D);
  arma::mat mean_indep(D, p);

  int offset = 0;
  for (int d = 0; d < D; d++) {
    int nd = n_d(d);
    arma::vec y_d = y_transformed.subvec(offset, offset + nd - 1);
    arma::mat X_d = X.rows(offset, offset + nd - 1);
    arma::vec w_d = weights.subvec(offset, offset + nd - 1);

    double wsum = arma::accu(w_d);

    // Weighted mean of dependent variable
    mean_dep(d) = arma::dot(w_d, y_d) / wsum;

    // Weighted means of independent variables
    for (int k = 0; k < p; k++) {
      mean_indep(d, k) = arma::dot(w_d, X_d.col(k)) / wsum;
    }

    delta2(d) = arma::dot(w_d, w_d) / (wsum * wsum);
    gamma_weight(d) = sigma2_u / (sigma2_u + sigma2_e * delta2(d));

    // Weighted least squares accumulation
    arma::mat W_diag = arma::diagmat(w_d);
    arma::vec dep_var_ast = y_d - gamma_weight(d) * mean_dep(d);
    arma::mat indep_weight = X_d.t() * W_diag;

    arma::mat indep_var_ast(nd, p);
    for (int i = 0; i < nd; i++) {
      for (int k = 0; k < p; k++) {
        indep_var_ast(i, k) = X_d(i, k) - gamma_weight(d) * mean_indep(d, k);
      }
    }

    num += indep_weight * dep_var_ast;
    den += indep_weight * indep_var_ast;

    offset += nd;
  }

  // Weighted betas
  arma::vec betas = arma::solve(den, num);

  // Weighted random effects
  arma::vec rand_eff(D);
  for (int d = 0; d < D; d++) {
    arma::rowvec mi = mean_indep.row(d);
    rand_eff(d) = gamma_weight(d) * (mean_dep(d) - arma::dot(mi.t(), betas));
  }

  return Rcpp::List::create(
    Rcpp::Named("betas") = betas,
    Rcpp::Named("rand_eff") = rand_eff,
    Rcpp::Named("gammaw") = gamma_weight,
    Rcpp::Named("delta2") = delta2
  );
}
```

- [ ] **Step 4: Compile and run tests**

```bash
Rscript -e 'Rcpp::compileAttributes(".")'
Rscript -e 'pkgbuild::compile_dll(".")'
Rscript -e 'devtools::test(filter = "lme_fit_cpp")'
```

- [ ] **Step 5: Commit**

```bash
git add src/lme_fit.cpp tests/testthat/test_lme_fit_cpp.R R/RcppExports.R src/RcppExports.cpp
git commit -m "feat: add model_par_weighted_cpp for pseudo-EB with survey weights"
```

---

### Task 4: Full Parametric Bootstrap in C++

**Files:**
- Create: `src/parametric_bootstrap.cpp`
- Create: `tests/testthat/test_bootstrap_cpp.R`

This is the main orchestration function. It runs the entire B-iteration loop in C++.

- [ ] **Step 1: Write failing test**

Create `tests/testthat/test_bootstrap_cpp.R`:
```r
test_that("parametric_bootstrap_cpp produces valid MSE (unweighted, log)", {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  set.seed(42)
  result <- ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    L = 5, MSE = TRUE, B = 3, transformation = "log"
  )

  mse <- estimators(result, indicator = "all", MSE = TRUE)
  mse_cols <- grep("_MSE$", names(mse$ind), value = TRUE)
  for (col in mse_cols) {
    expect_true(all(mse$ind[[col]] >= 0), info = paste("MSE:", col))
  }
})

test_that("parametric_bootstrap_cpp produces valid MSE (unweighted, box.cox)", {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  set.seed(42)
  result <- ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    L = 5, MSE = TRUE, B = 3, transformation = "box.cox"
  )

  mse <- estimators(result, indicator = "all", MSE = TRUE)
  mse_cols <- grep("_MSE$", names(mse$ind), value = TRUE)
  for (col in mse_cols) {
    expect_true(all(mse$ind[[col]] >= 0), info = paste("MSE:", col))
  }
})

test_that("parametric_bootstrap_cpp works with aggregate_to", {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  eusilcA_pop$region <- as.integer(as.factor(eusilcA_pop$district)) %% 5 + 1

  set.seed(42)
  result <- ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    L = 5, MSE = TRUE, B = 3,
    aggregate_to = "region"
  )

  mse <- estimators(result, indicator = "all", MSE = TRUE)
  expect_true(nrow(mse$ind) == 5)
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
Rscript -e 'devtools::test(filter = "bootstrap_cpp")'
```

- [ ] **Step 3: Implement parametric_bootstrap_cpp**

Create `src/parametric_bootstrap.cpp`. This is the largest single file. It calls functions from other translation units via forward declarations.

```cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Forward declarations from other translation units
Rcpp::NumericVector back_transform_cpp(const arma::vec& y,
                                        const std::string& transformation,
                                        double lambda, double shift);
arma::vec compute_domain_indicators_cpp(const arma::vec& y,
                                         const arma::vec& weights,
                                         double threshold);
arma::vec std_transform_y_cpp(const arma::vec& y_raw,
                               const std::string& transformation,
                               double lambda);
Rcpp::List data_transform_cpp(const arma::vec& y_raw,
                               const std::string& transformation,
                               double lambda);
Rcpp::List lme_fit_cpp(const arma::vec& y_transformed,
                        const arma::mat& X,
                        const arma::ivec& n_d);
Rcpp::List model_par_weighted_cpp(const arma::vec& y_transformed,
                                   const arma::mat& X,
                                   const arma::vec& weights,
                                   const arma::ivec& n_d,
                                   double sigma2_e, double sigma2_u);
double optimal_parameter_cpp(const arma::vec& y, const arma::mat& X,
                              const arma::ivec& domain_ids,
                              const arma::ivec& n_d,
                              const std::string& transformation,
                              double lower, double upper);

// Brent's method (local copy for this TU)
template <typename F>
static double brent_min(F f, double a, double b,
                         double tol = 1.490116e-08, int maxiter = 1000) {
  const double golden = 0.3819660112501051;
  double x = a + golden * (b - a), w = x, v = x;
  double fx = f(x), fw = fx, fv = fx, d = 0.0, e = 0.0;
  for (int iter = 0; iter < maxiter; ++iter) {
    double mid = 0.5 * (a + b), t1 = tol * std::abs(x) + 1e-10, t2 = 2.0 * t1;
    if (std::abs(x - mid) <= (t2 - 0.5 * (b - a))) return x;
    double p = 0, q = 0, r = 0;
    if (std::abs(e) > t1) {
      r = (x-w)*(fx-fv); q = (x-v)*(fx-fw); p = (x-v)*q-(x-w)*r;
      q = 2*(q-r); if (q>0) p=-p; else q=-q; r=e; e=d;
    }
    if (std::abs(p)<std::abs(.5*q*r) && p>q*(a-x) && p<q*(b-x)) {
      d=p/q; double u=x+d;
      if ((u-a)<t2||(b-u)<t2) d=(x<mid)?t1:-t1;
    } else { e=(x<mid)?b-x:a-x; d=golden*e; }
    double u=(std::abs(d)>=t1)?x+d:x+((d>0)?t1:-t1), fu=f(u);
    if (fu<=fx) { if(u<x) b=x; else a=x; v=w;fv=fw;w=x;fw=fx;x=u;fx=fu; }
    else { if(u<x)a=u;else b=u;
      if(fu<=fw||w==x){v=w;fv=fw;w=u;fw=fu;}
      else if(fu<=fv||v==x||v==w){v=u;fv=fu;} }
  }
  return x;
}

// ---------------------------------------------------------------------------
// parametric_bootstrap_cpp: Full B-iteration parametric bootstrap in C++.
//
// Runs B iterations of:
//   1. Generate superpopulation
//   2. Compute true indicators on superpop
//   3. Generate bootstrap sample
//   4. Find optimal lambda (if needed)
//   5. Transform bootstrap sample
//   6. Fit LME model (C++ REML)
//   7. Compute model_par (unweighted or weighted)
//   8. Compute gen_model (sigmav2, mu)
//   9. Run Monte Carlo simulation
//   10. Accumulate MSE
//
// Returns: MSE matrix [N_dom_ind x 10]
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat parametric_bootstrap_cpp(
    // Population data
    const arma::mat& X_pop,
    const arma::vec& mu_fixed_orig,
    const arma::ivec& n_pop,
    const arma::ivec& obs_dom,
    const arma::ivec& dist_obs_dom,
    const arma::vec& pop_weights,
    int N_pop, int N_dom_pop,
    // Sample data
    const arma::mat& X_smp,
    const arma::vec& y_smp_orig,
    const arma::ivec& n_smp,
    const arma::ivec& smp_domain_ids,
    const arma::ivec& smp_to_pop_map,
    int N_smp, int N_dom_smp,
    // Original model parameters
    double sigmae2_orig,
    double sigmau2_orig,
    const arma::vec& sigmav2_orig,
    int N_dom_smp_selected, int N_dom_unobs,
    // Algorithm parameters
    int B, int L,
    double threshold,
    const std::string& transformation,
    double interval_lower, double interval_upper,
    // Aggregate domain info (optional)
    const Rcpp::Nullable<Rcpp::IntegerVector>& agg_domain_ids_pop,
    int N_dom_agg,
    // Sample weights (optional, NULL for unweighted)
    const Rcpp::Nullable<Rcpp::NumericVector>& smp_weights_nullable
) {
  // Determine indicator domain count
  bool use_agg = agg_domain_ids_pop.isNotNull() && N_dom_agg > 0;
  int N_dom_ind = use_agg ? N_dom_agg : N_dom_pop;

  // Agg domain IDs for population
  arma::ivec agg_ids_pop;
  std::vector<arma::uvec> agg_idx_cache;
  if (use_agg) {
    agg_ids_pop = Rcpp::as<arma::ivec>(agg_domain_ids_pop.get());
    agg_idx_cache.resize(N_dom_ind);
    for (int d = 0; d < N_dom_ind; d++)
      agg_idx_cache[d] = arma::find(agg_ids_pop == (d + 1));
  }

  // Sample weights
  bool weighted = smp_weights_nullable.isNotNull();
  arma::vec smp_weights_vec;
  if (weighted) {
    smp_weights_vec = Rcpp::as<arma::vec>(smp_weights_nullable.get());
  }

  // Whether lambda needs optimization
  bool needs_lambda_opt = (transformation != "no" && transformation != "log");

  // Pre-compute sqrt of original variances
  double sqrt_sigmae2 = std::sqrt(sigmae2_orig);
  double sqrt_sigmau2 = std::sqrt(sigmau2_orig);
  double sqrt_combined = std::sqrt(sigmae2_orig + sigmau2_orig);

  int n_indicators = 10;
  arma::mat mse_accum(N_dom_ind, n_indicators, arma::fill::zeros);

  for (int b = 0; b < B; b++) {
    // ================================================================
    // Step 1: Generate superpopulation
    // ================================================================
    arma::vec eps_pop(N_pop);
    // In-sample obs get N(0, sigma2_e), out-of-sample get N(0, sigma2_e + sigma2_u)
    for (int i = 0; i < N_pop; i++) {
      if (obs_dom(i)) {
        eps_pop(i) = R::rnorm(0.0, sqrt_sigmae2);
      } else {
        eps_pop(i) = R::rnorm(0.0, sqrt_combined);
      }
    }
    arma::vec vu_tmp(N_dom_pop);
    for (int d = 0; d < N_dom_pop; d++)
      vu_tmp(d) = R::rnorm(0.0, sqrt_sigmau2);
    arma::vec vu_pop(N_pop);
    {
      int off = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        for (int i = 0; i < n_pop(d); i++)
          vu_pop(off + i) = vu_tmp(d);
        off += n_pop(d);
      }
    }
    arma::vec Y_pop_notrans = mu_fixed_orig + eps_pop + vu_pop;
    Rcpp::NumericVector Y_pop_bt = back_transform_cpp(
      Y_pop_notrans, transformation,
      needs_lambda_opt ? 0.0 : 0.0,  // lambda not used for log/no
      0.0  // shift for superpop is always 0 (original model shift)
    );
    // WAIT: The superpopulation uses the ORIGINAL lambda and shift.
    // We need those passed in. Let me reconsider...
    // Actually: superpopulation generates on the transformed scale using
    // mu_fixed_orig (which is X_pop * original_betas on transformed scale),
    // then back-transforms. The back-transform needs the original lambda and shift.
    // These must be passed in as parameters.
    // I'll add lambda_orig and shift_orig to the function signature.

    // For now, placeholder — this will be corrected in implementation.
    // The actual code will receive lambda_orig and shift_orig and use them here.
    arma::vec Y_pop_b = Rcpp::as<arma::vec>(Y_pop_bt);
    for (int i = 0; i < N_pop; i++)
      if (!std::isfinite(Y_pop_b(i))) Y_pop_b(i) = 0.0;

    // ================================================================
    // Step 2: Compute true indicators on superpopulation
    // ================================================================
    arma::mat true_ind(N_dom_ind, n_indicators);
    if (use_agg) {
      for (int d = 0; d < N_dom_ind; d++) {
        arma::vec y_d = Y_pop_b.elem(agg_idx_cache[d]);
        arma::vec w_d = pop_weights.elem(agg_idx_cache[d]);
        true_ind.row(d) = compute_domain_indicators_cpp(y_d, w_d, threshold).t();
      }
    } else {
      int off = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        int nd = n_pop(d);
        arma::vec y_d = Y_pop_b.subvec(off, off + nd - 1);
        arma::vec w_d = pop_weights.subvec(off, off + nd - 1);
        true_ind.row(d) = compute_domain_indicators_cpp(y_d, w_d, threshold).t();
        off += nd;
      }
    }

    // ================================================================
    // Step 3: Generate bootstrap sample
    // ================================================================
    arma::vec eps_smp(N_smp);
    for (int i = 0; i < N_smp; i++)
      eps_smp(i) = R::rnorm(0.0, sqrt_sigmae2);

    arma::vec vu_smp(N_smp);
    {
      int off = 0;
      for (int d = 0; d < N_dom_smp; d++) {
        double vu_d;
        int pop_idx = smp_to_pop_map(d);
        if (pop_idx != NA_INTEGER && pop_idx > 0) {
          vu_d = vu_tmp(pop_idx - 1);
        } else {
          vu_d = R::rnorm(0.0, sqrt_sigmau2);
        }
        for (int i = 0; i < n_smp(d); i++)
          vu_smp(off + i) = vu_d;
        off += n_smp(d);
      }
    }

    // mu_smp = X_smp * original_betas (on transformed scale)
    // But we need the BACK-TRANSFORMED bootstrap sample y
    // Y_smp_b_trans = X_smp * betas_orig + eps + vu (on transformed scale)
    // Y_smp_b = back_transform(Y_smp_b_trans)
    // Actually, looking at bootstrap_par(): it computes on transformed scale
    // then back-transforms. The betas used are the ORIGINAL betas.
    // mu_smp = X_smp %*% model_par$betas (these are on the transformed scale)
    // We need the original betas passed in too.
    // Let me add betas_orig to the function signature.

    // Y_smp_b_trans = X_smp * betas_orig + eps + vu
    // Y_smp_b = back_transform(Y_smp_b_trans, transformation, lambda_orig, shift_orig)
    // Then: the UNTRANSFORMED Y_smp_b becomes the new sample response.

    // Placeholder for betas_orig usage — will be in final implementation.

    // ================================================================
    // Step 4: optimal_parameter on bootstrap sample
    // ================================================================
    // double lambda_b;
    // if (needs_lambda_opt) {
    //   lambda_b = optimal_parameter_cpp(Y_smp_untrans, X_smp, smp_domain_ids,
    //                                     n_smp, transformation,
    //                                     interval_lower, interval_upper);
    // } else {
    //   lambda_b = 0.0; // not used for log/no
    // }

    // ================================================================
    // Step 5-9: Transform, fit, gen_model, monte_carlo
    // ================================================================
    // ... (detailed in implementation) ...

    // ================================================================
    // Step 10: Accumulate MSE
    // ================================================================
    // mse_accum += (boot_estimates - true_ind) % (boot_estimates - true_ind);

    // Progress reporting
    if ((b + 1) % 10 == 0 && (b + 1) != B) {
      Rcpp::Rcout << "\r" << (b + 1) << " of " << B
                  << " Bootstrap iterations completed" << std::endl;
    }
  }

  return mse_accum / (double)B;
}
```

**NOTE TO IMPLEMENTER**: The code above is a **skeleton showing the structure**. The actual implementation must:
1. Add `lambda_orig`, `shift_orig`, and `betas_orig` to the function signature
2. Implement the complete superpopulation back-transform with correct lambda/shift
3. Implement the complete bootstrap sample generation and back-transform
4. Call `data_transform_cpp` on the back-transformed bootstrap sample
5. Call `optimal_parameter_cpp` or skip for log/no
6. Call `lme_fit_cpp` on the transformed bootstrap sample
7. Handle unweighted vs weighted via `model_par_weighted_cpp`
8. Compute gen_model parameters (sigmav2, mu, mu_fixed) from the fit
9. Run `monte_carlo_cpp` (forward-declared from monte_carlo.cpp)
10. Accumulate squared differences
11. Print progress every 10 iterations

The key data flow is:
```
superpop (transformed scale) → back_transform → Y_pop (original scale) → true indicators
bootstrap_par (transformed scale) → back_transform → Y_smp (original scale)
Y_smp → data_transform → Y_smp_trans → lme_fit → model params
model params → gen_model → mu, sigmav2 → monte_carlo → boot estimates
MSE += (boot_estimates - true_indicators)^2
```

- [ ] **Step 4: Compile and run tests**

```bash
Rscript -e 'Rcpp::compileAttributes(".")'
Rscript -e 'pkgbuild::compile_dll(".")'
Rscript -e 'devtools::test(filter = "bootstrap_cpp")'
```

- [ ] **Step 5: Commit**

```bash
git add src/parametric_bootstrap.cpp tests/testthat/test_bootstrap_cpp.R R/RcppExports.R src/RcppExports.cpp
git commit -m "feat: add full C++ parametric bootstrap loop"
```

---

### Task 5: Wire R parametric_bootstrap() to C++

**Files:**
- Modify: `R/mse_estimation.R`

Add dispatch logic at the top of `parametric_bootstrap()`: if `boot_type == "parametric"` and no custom indicators (standard 10 only), call `parametric_bootstrap_cpp()` with all the extracted framework vectors. Otherwise fall back to existing R loop.

- [ ] **Step 1: Modify parametric_bootstrap()**

In `R/mse_estimation.R`, add the C++ dispatch at the beginning of the function, after the `boot_type == "wild"` check:

```r
parametric_bootstrap <- function(framework, point_estim, fixed,
                                 transformation, interval = c(-1, 2),
                                 L, B, boot_type, parallel_mode, cpus,
                                 control, true_indicators) {
  message("\r", "Bootstrap started                                            ")

  # Check if we can use the C++ fast path
  n_custom <- length(framework$indicator_names) - 10
  use_cpp <- (boot_type == "parametric" && n_custom == 0 &&
              is.null(true_indicators) && cpus <= 1)

  if (use_cpp) {
    # Prepare all vectors for C++
    # ... (extract from framework and point_estim)
    # Call parametric_bootstrap_cpp(...)
    # Format result and return
  }

  # ... existing R code as fallback ...
}
```

The R wrapper extracts all needed vectors from the `framework` and `point_estim` objects and passes them as flat vectors/matrices to C++.

- [ ] **Step 2: Run full test suite**

```bash
Rscript -e 'devtools::test()'
```

Expected: ALL tests pass.

- [ ] **Step 3: Commit**

```bash
git add R/mse_estimation.R
git commit -m "feat: wire parametric_bootstrap() to C++ fast path"
```

---

### Task 6: Benchmark and Final Verification

**Files:** No new files.

- [ ] **Step 1: Benchmark emdi vs emdi2 with MSE**

```bash
Rscript -e '
library(devtools); load_all(".")
data("eusilcA_smp"); data("eusilcA_pop")
fixed <- eqIncome ~ gender + eqsize + cash + self_empl +
  unempl_ben + age_ben + surv_ben + sick_ben + dis_ben +
  rent + fam_allow + house_allow + cap_inv + tax_adj

cat("--- L=50, MSE=TRUE, B=50 ---\n")
set.seed(42)
t_orig <- system.time(emdi::ebp(fixed=fixed, pop_data=eusilcA_pop,
  pop_domains="district", smp_data=eusilcA_smp, smp_domains="district",
  L=50, MSE=TRUE, B=50))
cat("emdi  (pure R):", t_orig["elapsed"], "s\n")

set.seed(42)
t_cpp <- system.time(ebp(fixed=fixed, pop_data=eusilcA_pop,
  pop_domains="district", smp_data=eusilcA_smp, smp_domains="district",
  L=50, MSE=TRUE, B=50))
cat("emdi2 (C++):   ", t_cpp["elapsed"], "s\n")
cat("Speedup:       ", round(t_orig["elapsed"]/t_cpp["elapsed"], 1), "x\n")
'
```

- [ ] **Step 2: Run full test suite**

```bash
Rscript -e 'devtools::test()'
```

- [ ] **Step 3: Push**

```bash
git push origin dev-emdi-reml-cpp
```

---

## Important Implementation Notes

### RNG Order

The C++ bootstrap must consume random numbers in the same order as the R code for reproducibility within a single run. However, since the C++ and R paths use different model fitting (C++ REML vs nlme), the results won't be bit-identical between C++ and R fallback paths — and that's expected. What matters is that the C++ path produces **statistically equivalent** MSE estimates.

### Domain Matching

The `smp_to_pop_map` vector maps each sample domain index to its position in the population domain vector (1-based, NA if not found). This handles `selected_domains` where some sample domains may not be in the selected population domains.

### gen_model in C++

The gen_model computation inside the bootstrap loop is:
```
gamma_d = sigma2_u / (sigma2_u + sigma2_e / n_smp_d)      [unweighted]
gamma_d = gammaw_d                                          [weighted]
sigmav2_d = sigma2_u * (1 - gamma_d)
rand_eff_pop = rep(rand_eff, n_pop)  [matched by domain]
mu_fixed = X_pop * betas
mu = mu_fixed + rand_eff_pop
```

### Threshold as function

The R code supports `threshold` as a function (`inherits(framework$threshold, "function")`). In the C++ path, the threshold should be pre-computed before calling C++. The R wrapper handles this.
