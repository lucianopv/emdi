# C++ REML Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `optimal_parameter()` -> `lme()` chain (72% of MSE bootstrap time) with a direct C++ REML log-likelihood computation using closed-form per-domain sufficient statistics.

**Architecture:** A single exported C++ function `optimal_parameter_cpp()` performs both the outer optimization over lambda (transformation parameter) and the inner optimization over theta = sigma2_u/sigma2_e (variance ratio). Both use Brent's method entirely in C++. The REML log-likelihood is computed analytically from per-domain sufficient statistics using the Woodbury identity for the block-diagonal covariance.

**Tech Stack:** R, Rcpp, RcppArmadillo, Brent's method (R's `R_zeroin2` C API)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/reml_optimization.cpp` | Create | Standardized transforms, REML profile likelihood, Brent optimization for theta and lambda |
| `R/optimal_parameter.R` | Modify | Call `optimal_parameter_cpp()` instead of R `optimize()` -> `lme()` chain |
| `tests/testthat/test_reml_cpp.R` | Create | Tests verifying C++ matches R for all transformation types |

---

### Task 1: Standardized Forward Transformations in C++

**Files:**
- Create: `src/reml_optimization.cpp`
- Create: `tests/testthat/test_reml_cpp.R`

The standardized transformations scale the data so that the REML log-likelihood is comparable across lambda values. These are different from the back-transformations already in C++ — these are the *forward* scaled transforms used during optimization.

- [ ] **Step 1: Write failing tests for standardized transformations**

Create `tests/testthat/test_reml_cpp.R`:
```r
test_that("std_transform_y_cpp matches R box_cox_std", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))

  for (lam in c(0.2, 0.5, 0.7, 1.0, 1.5, -0.5)) {
    r_result <- box_cox_std(y, lam)
    cpp_result <- std_transform_y_cpp(y, "box.cox", lam)
    expect_equal(as.numeric(cpp_result), as.numeric(r_result),
                 tolerance = 1e-10,
                 info = paste("box.cox lambda =", lam))
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
                 tolerance = 1e-10,
                 info = paste("dual lambda =", lam))
  }

  # lambda ~ 0
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
                 tolerance = 1e-10,
                 info = paste("log.shift lambda =", lam))
  }
})

test_that("std_transform_y_cpp handles negative values with shift", {
  y <- c(-5, -2, 0, 3, 10, 50)
  r_result <- box_cox_std(y, 0.5)
  cpp_result <- std_transform_y_cpp(y, "box.cox", 0.5)
  expect_equal(as.numeric(cpp_result), as.numeric(r_result), tolerance = 1e-10)
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/home/lpv/Dropbox/Inglaterra_2019/PhD/Southampton/R Packages/emdi"
Rscript -e 'devtools::test(filter = "reml_cpp")'
```

Expected: FAIL with "could not find function std_transform_y_cpp"

- [ ] **Step 3: Implement std_transform_y_cpp in src/reml_optimization.cpp**

```cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// ---------------------------------------------------------------------------
// Standardized forward transformations for REML optimization.
// These transform y so that the REML log-likelihood is comparable across
// lambda values. Must match R functions: box_cox_std(), dual_std(),
// log_shift_opt_std() in R/transformation_functions.R.
// ---------------------------------------------------------------------------

static double geometric_mean(const arma::vec& x) {
  return std::exp(arma::mean(arma::log(x)));
}

// [[Rcpp::export]]
arma::vec std_transform_y_cpp(const arma::vec& y_raw,
                               const std::string& transformation,
                               double lambda) {
  int n = y_raw.n_elem;
  arma::vec y = y_raw;  // working copy

  if (transformation == "box.cox") {
    // Shift if min <= 0
    double mn = y.min();
    if (mn <= 0) {
      y = y - mn + 1.0;
    }
    double gm = geometric_mean(y);

    arma::vec result(n);
    if (std::abs(lambda) > 1e-12) {
      // (y^lambda - 1) / (lambda * gm^(lambda-1))
      double scale = lambda * std::pow(gm, lambda - 1.0);
      result = (arma::pow(y, lambda) - 1.0) / scale;
    } else {
      // gm * log(y)
      result = gm * arma::log(y);
    }
    return result;

  } else if (transformation == "dual") {
    // Shift if min <= 0
    double mn = y.min();
    if (mn <= 0) {
      y = y - mn + 1.0;
    }

    if (std::abs(lambda) > 1e-12) {
      // yt = (y^lambda - y^(-lambda)) / (2*lambda)
      arma::vec yt = (arma::pow(y, lambda) - arma::pow(y, -lambda)) / (2.0 * lambda);
      // geo = geometric_mean(y^(lambda-1) + y^(-lambda-1))
      double geo = geometric_mean(arma::pow(y, lambda - 1.0) + arma::pow(y, -lambda - 1.0));
      return yt * 2.0 / geo;
    } else {
      // gm * log(y)
      double gm = geometric_mean(y);
      return gm * arma::log(y);
    }

  } else if (transformation == "log.shift") {
    // Adjust lambda if min(y + lambda) <= 0
    double mn = arma::min(y + lambda);
    if (mn <= 0) {
      lambda = lambda + std::abs(y.min()) + 1.0;
    }
    // gm = geometric_mean(y + lambda)
    arma::vec ypl = y + lambda;
    double gm = geometric_mean(ypl);
    return gm * arma::log(ypl);

  } else {
    Rcpp::stop("Unknown transformation for std_transform: " + transformation);
    return y; // unreachable
  }
}
```

- [ ] **Step 4: Compile and run tests**

```bash
Rscript -e 'Rcpp::compileAttributes(".")'
Rscript -e 'pkgbuild::compile_dll(".")'
Rscript -e 'devtools::test(filter = "reml_cpp")'
```

Expected: All standardized transform tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reml_optimization.cpp tests/testthat/test_reml_cpp.R R/RcppExports.R src/RcppExports.cpp
git commit -m "feat: add C++ standardized forward transformations for REML optimization"
```

---

### Task 2: Profile REML Log-Likelihood in C++

**Files:**
- Modify: `src/reml_optimization.cpp`
- Modify: `tests/testthat/test_reml_cpp.R`

Implement the core REML computation: given a transformed y vector and design matrix X, compute the profile REML negative log-likelihood by optimizing over theta = sigma2_u/sigma2_e using Brent's method.

- [ ] **Step 1: Write failing test for REML loglik**

Append to `tests/testthat/test_reml_cpp.R`:
```r
test_that("reml_loglik_cpp matches lme() REML log-likelihood", {
  data("eusilcA_smp", package = "emdi2")
  fixed <- eqIncome ~ gender + eqsize

  # Test with box.cox at several lambda values
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

    y <- as.numeric(eusilcA_smp$eqIncome)
    X <- model.matrix(fixed, eusilcA_smp)
    domain_ids <- as.integer(as.factor(eusilcA_smp$district))
    n_d <- as.integer(table(as.factor(eusilcA_smp$district)))

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

  y <- as.numeric(eusilcA_smp$eqIncome)
  X <- model.matrix(fixed, eusilcA_smp)
  domain_ids <- as.integer(as.factor(eusilcA_smp$district))
  n_d <- as.integer(table(as.factor(eusilcA_smp$district)))

  cpp_nll <- reml_loglik_cpp(
    lambda = lam, y = y, X = X,
    domain_ids = domain_ids, n_d = n_d,
    transformation = "dual"
  )
  expect_equal(cpp_nll, lme_nll, tolerance = 1e-4)
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
Rscript -e 'devtools::test(filter = "reml_cpp")'
```

Expected: FAIL with "could not find function reml_loglik_cpp"

- [ ] **Step 3: Implement reml_profile_eval and reml_loglik_cpp**

Add to `src/reml_optimization.cpp`:

```cpp
// ---------------------------------------------------------------------------
// Profile REML log-likelihood evaluation at a given theta = sigma2_u/sigma2_e.
//
// For the nested error model y = Xb + Zu + e with:
//   u_i ~ N(0, sigma2_u), e_ij ~ N(0, sigma2_e)
//
// V_d = sigma2_e * I + sigma2_u * J  (block diagonal)
// V_d^{-1}/sigma2_e = I - c_d * J  where c_d = theta/(1 + n_d*theta)
//
// The sufficient statistics per domain are precomputed once:
//   S_xx_d = X_d' X_d,  S_xy_d = X_d' y_d,  S_yy_d = y_d' y_d
//   xbar_d = colSums(X_d),  ybar_d = sum(y_d)
//
// Given theta, the accumulated weighted statistics are:
//   A = sum_d [ S_xx_d - c_d * xbar_d * xbar_d' ]   (= X'V^{-1}X / sigma2_e)
//   b = sum_d [ S_xy_d - c_d * xbar_d * ybar_d ]     (= X'V^{-1}y / sigma2_e)
//
// Then: beta = A^{-1} b
//       quad = sum_d [ (y_d - X_d*beta)'(I - c_d*J)(y_d - X_d*beta) ]
//       sigma2_e_hat = quad / (n - p)
//       sigma2_u_hat = theta * sigma2_e_hat
//
// REML loglik = -0.5 * [(n-p)*log(2pi) + log|V| + log|X'V^{-1}X| + quad_actual]
// ---------------------------------------------------------------------------

struct DomainStats {
  arma::mat S_xx;    // X_d' X_d  (p x p)
  arma::vec S_xy;    // X_d' y_d  (p x 1)
  double S_yy;       // y_d' y_d
  arma::vec xbar;    // colSums(X_d)  (p x 1)
  double ybar;       // sum(y_d)
  int n_d;           // domain size
};

// Evaluate -REML loglik at given theta, using precomputed domain stats.
// Returns negative REML log-likelihood.
static double reml_profile_eval(double log_theta,
                                 const std::vector<DomainStats>& stats,
                                 int n, int p, int D) {
  double theta = std::exp(log_theta);

  // Accumulate A = X'V^{-1}X / sigma2_e  and  b = X'V^{-1}y / sigma2_e
  arma::mat A(p, p, arma::fill::zeros);
  arma::vec b(p, arma::fill::zeros);
  double log_det_ratio = 0.0;  // sum of log(1 + n_d * theta)

  for (int d = 0; d < D; d++) {
    const DomainStats& ds = stats[d];
    double c_d = theta / (1.0 + ds.n_d * theta);

    A += ds.S_xx - c_d * (ds.xbar * ds.xbar.t());
    b += ds.S_xy - c_d * ds.xbar * ds.ybar;
    log_det_ratio += std::log(1.0 + ds.n_d * theta);
  }

  // Solve for beta_hat
  arma::vec beta_hat = arma::solve(A, b, arma::solve_opts::likely_sympd);

  // Compute quadratic form: r'V^{-1}r / sigma2_e
  // = sum_d [ (y_d - X_d*beta)'(I - c_d*J)(y_d - X_d*beta) ]
  // = sum_d [ S_yy_d - 2*beta'*S_xy_d + beta'*S_xx_d*beta
  //           - c_d*(ybar_d - xbar_d'*beta)^2 ]
  double quad = 0.0;
  for (int d = 0; d < D; d++) {
    const DomainStats& ds = stats[d];
    double c_d = theta / (1.0 + ds.n_d * theta);

    double rss_d = ds.S_yy
      - 2.0 * arma::dot(beta_hat, ds.S_xy)
      + arma::as_scalar(beta_hat.t() * ds.S_xx * beta_hat);
    double rbar_d = ds.ybar - arma::dot(ds.xbar, beta_hat);
    quad += rss_d - c_d * rbar_d * rbar_d;
  }

  // Profile sigma2_e
  double sigma2_e = quad / (double)(n - p);

  // Full log-determinant of V
  // log|V| = sum_d [(n_d-1)*log(sigma2_e) + log(sigma2_e + n_d*sigma2_u)]
  //        = (n - D)*log(sigma2_e) + sum_d log(sigma2_e + n_d*theta*sigma2_e)
  //        = (n - D)*log(sigma2_e) + sum_d [log(sigma2_e) + log(1 + n_d*theta)]
  //        = n*log(sigma2_e) - D*log(sigma2_e) + D*log(sigma2_e) + log_det_ratio
  //  Wait, let me be precise:
  //  log|V_d| = (n_d - 1)*log(sigma2_e) + log(sigma2_e + n_d*sigma2_u)
  //           = (n_d - 1)*log(sigma2_e) + log(sigma2_e) + log(1 + n_d*theta)
  //           = n_d*log(sigma2_e) + log(1 + n_d*theta)
  //  log|V| = sum_d [n_d*log(sigma2_e) + log(1 + n_d*theta)]
  //         = n*log(sigma2_e) + log_det_ratio
  double log_det_V = n * std::log(sigma2_e) + log_det_ratio;

  // log|X'V^{-1}X| = log|A / sigma2_e| ... wait, A = X'V^{-1}X / sigma2_e
  // so X'V^{-1}X = A * sigma2_e ... no, A IS X'V^{-1}X/sigma2_e
  // Actually: X'V^{-1}X = A / sigma2_e?  No.
  // V^{-1} = (1/sigma2_e)(I - c_d*J) so X'V^{-1}X = (1/sigma2_e) * A_raw
  // where A_raw = sum_d [X_d'(I - c_d*J)X_d] = A (what we computed).
  // So X'V^{-1}X = A / sigma2_e... No!
  // We computed A = sum_d [S_xx_d - c_d * xbar*xbar'] which is X'(V^{-1}/sigma2_e)X = A
  // So X'V^{-1}X = A / sigma2_e.  Wait:
  // V_d^{-1} = (1/sigma2_e)(I - c_d J)
  // X_d' V_d^{-1} X_d = (1/sigma2_e)(X_d'X_d - c_d * xbar*xbar')
  //                    = (1/sigma2_e) * A_d
  // So A = sum A_d, and X'V^{-1}X = A / sigma2_e. No wait:
  // A = sum_d A_d = sum_d (S_xx_d - c_d * xbar*xbar')
  // X'V^{-1}X = (1/sigma2_e) * A
  // log|X'V^{-1}X| = log|(1/sigma2_e)*A| = -p*log(sigma2_e) + log|A|

  double log_det_A;
  double sign;
  arma::log_det(log_det_A, sign, A);
  double log_det_XtVinvX = -p * std::log(sigma2_e) + log_det_A;

  // quad_actual = r'V^{-1}r = quad / sigma2_e = (n-p)*sigma2_e / sigma2_e = (n-p)
  // Since sigma2_e = quad/(n-p), quad_actual = quad/sigma2_e = (n-p)
  double quad_actual = (double)(n - p);

  // REML log-likelihood
  double reml_ll = -0.5 * ((n - p) * std::log(2.0 * M_PI) +
                            log_det_V + log_det_XtVinvX + quad_actual);

  return -reml_ll;  // return NEGATIVE log-likelihood (for minimization)
}

// ---------------------------------------------------------------------------
// Brent's method for 1D minimization on [a, b].
// Adapted from R's C API (Brent 1973). Tolerance matches R's optimize().
// ---------------------------------------------------------------------------
static double brent_minimize(
    double ax, double bx,
    std::function<double(double)> f,
    double tol = 1.4901161193847656e-08,  // sqrt(.Machine$double.eps)
    int maxiter = 1000
) {
  // Golden ratio
  const double c = 0.5 * (3.0 - std::sqrt(5.0));

  double a = ax, b = bx;
  double x = a + c * (b - a);
  double w = x, v = x;
  double fx = f(x), fw = fx, fv = fx;
  double d = 0.0, e = 0.0;

  for (int iter = 0; iter < maxiter; iter++) {
    double midpoint = 0.5 * (a + b);
    double tol1 = tol * std::abs(x) + 1e-10;
    double tol2 = 2.0 * tol1;

    if (std::abs(x - midpoint) <= (tol2 - 0.5 * (b - a))) {
      return x;
    }

    double p_val = 0, q_val = 0, r_val = 0;
    if (std::abs(e) > tol1) {
      r_val = (x - w) * (fx - fv);
      q_val = (x - v) * (fx - fw);
      p_val = (x - v) * q_val - (x - w) * r_val;
      q_val = 2.0 * (q_val - r_val);
      if (q_val > 0) p_val = -p_val;
      else q_val = -q_val;
      r_val = e;
      e = d;
    }

    if (std::abs(p_val) < std::abs(0.5 * q_val * r_val) &&
        p_val > q_val * (a - x) && p_val < q_val * (b - x)) {
      d = p_val / q_val;
      double u = x + d;
      if ((u - a) < tol2 || (b - u) < tol2) {
        d = (x < midpoint) ? tol1 : -tol1;
      }
    } else {
      e = (x < midpoint) ? b - x : a - x;
      d = c * e;
    }

    double u;
    if (std::abs(d) >= tol1) u = x + d;
    else u = x + ((d > 0) ? tol1 : -tol1);

    double fu = f(u);

    if (fu <= fx) {
      if (u < x) b = x; else a = x;
      v = w; fv = fw;
      w = x; fw = fx;
      x = u; fx = fu;
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w; fv = fw;
        w = u; fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u; fv = fu;
      }
    }
  }
  return x;
}

// ---------------------------------------------------------------------------
// reml_loglik_cpp: Compute negative REML log-likelihood at a given lambda.
//
// 1. Apply standardized transformation to y
// 2. Precompute per-domain sufficient statistics
// 3. Optimize over log(theta) using Brent's method
// 4. Return negative REML log-likelihood at optimal theta
//
// [[Rcpp::export]]
double reml_loglik_cpp(double lambda,
                        const arma::vec& y,
                        const arma::mat& X,
                        const arma::ivec& domain_ids,
                        const arma::ivec& n_d,
                        const std::string& transformation) {
  int n = y.n_elem;
  int p = X.n_cols;
  int D = n_d.n_elem;

  // Step 1: Standardized transformation
  arma::vec y_std = std_transform_y_cpp(y, transformation, lambda);

  // Step 2: Precompute per-domain sufficient statistics
  std::vector<DomainStats> stats(D);
  int offset = 0;
  for (int d = 0; d < D; d++) {
    int nd = n_d(d);
    stats[d].n_d = nd;

    arma::vec y_d = y_std.subvec(offset, offset + nd - 1);
    arma::mat X_d = X.rows(offset, offset + nd - 1);

    stats[d].S_xx = X_d.t() * X_d;
    stats[d].S_xy = X_d.t() * y_d;
    stats[d].S_yy = arma::dot(y_d, y_d);
    stats[d].xbar = arma::sum(X_d, 0).t();  // column sums
    stats[d].ybar = arma::accu(y_d);

    offset += nd;
  }

  // Step 3: Optimize over log(theta) using Brent's method
  // Search interval: log(theta) in [-15, 15] covers theta in [3e-7, 3e6]
  auto objective = [&](double log_theta) {
    return reml_profile_eval(log_theta, stats, n, p, D);
  };

  double optimal_log_theta = brent_minimize(-15.0, 15.0, objective);

  // Return the negative REML loglik at optimal theta
  return reml_profile_eval(optimal_log_theta, stats, n, p, D);
}
```

- [ ] **Step 4: Compile and run tests**

```bash
Rscript -e 'Rcpp::compileAttributes(".")'
Rscript -e 'pkgbuild::compile_dll(".")'
Rscript -e 'devtools::test(filter = "reml_cpp")'
```

Expected: All tests PASS, including the lme comparison tests (within tolerance 1e-4).

- [ ] **Step 5: Run full test suite**

```bash
Rscript -e 'devtools::test()'
```

Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/reml_optimization.cpp tests/testthat/test_reml_cpp.R R/RcppExports.R src/RcppExports.cpp
git commit -m "feat: add C++ profile REML log-likelihood with Brent optimization"
```

---

### Task 3: Full optimal_parameter_cpp

**Files:**
- Modify: `src/reml_optimization.cpp`
- Modify: `tests/testthat/test_reml_cpp.R`

Add the top-level function that optimizes over lambda, replacing the entire R chain.

- [ ] **Step 1: Write failing test**

Append to `tests/testthat/test_reml_cpp.R`:
```r
test_that("optimal_parameter_cpp matches R optimal_parameter for box.cox", {
  data("eusilcA_smp", package = "emdi2")
  fixed <- eqIncome ~ gender + eqsize

  # R version
  r_lambda <- optimal_parameter(
    generic_opt = generic_opt,
    fixed = fixed,
    smp_data = eusilcA_smp,
    smp_domains = "district",
    transformation = "box.cox",
    interval = "default",
    control = list()
  )

  # C++ version
  y <- as.numeric(eusilcA_smp$eqIncome)
  X <- model.matrix(fixed, eusilcA_smp)
  smp_data_sorted <- eusilcA_smp[order(eusilcA_smp$district), ]
  domain_ids <- as.integer(as.factor(smp_data_sorted$district))
  n_d <- as.integer(table(as.factor(smp_data_sorted$district)))
  y_sorted <- as.numeric(smp_data_sorted$eqIncome)
  X_sorted <- model.matrix(fixed, smp_data_sorted)

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
Rscript -e 'devtools::test(filter = "reml_cpp")'
```

Expected: FAIL with "could not find function optimal_parameter_cpp"

- [ ] **Step 3: Implement optimal_parameter_cpp**

Add to `src/reml_optimization.cpp`:

```cpp
// ---------------------------------------------------------------------------
// optimal_parameter_cpp: Find optimal transformation parameter lambda
// by minimizing the negative REML log-likelihood over the given interval.
//
// This replaces the entire R chain:
//   optimal_parameter() -> optimize() -> generic_opt() -> reml() -> lme()
//
// IMPORTANT: y and X must be sorted by domain (matching domain_ids order).
// domain_ids must be contiguous integers 1..D.
//
// [[Rcpp::export]]
double optimal_parameter_cpp(const arma::vec& y,
                              const arma::mat& X,
                              const arma::ivec& domain_ids,
                              const arma::ivec& n_d,
                              const std::string& transformation,
                              double lower,
                              double upper) {
  auto objective = [&](double lambda) {
    return reml_loglik_cpp(lambda, y, X, domain_ids, n_d, transformation);
  };

  double optimal_lambda = brent_minimize(lower, upper, objective);
  return optimal_lambda;
}
```

- [ ] **Step 4: Compile and run tests**

```bash
Rscript -e 'Rcpp::compileAttributes(".")'
Rscript -e 'pkgbuild::compile_dll(".")'
Rscript -e 'devtools::test(filter = "reml_cpp")'
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reml_optimization.cpp tests/testthat/test_reml_cpp.R R/RcppExports.R src/RcppExports.cpp
git commit -m "feat: add optimal_parameter_cpp - full lambda optimization in C++"
```

---

### Task 4: Wire R optimal_parameter() to C++

**Files:**
- Modify: `R/optimal_parameter.R`
- Modify: `tests/testthat/test_reml_cpp.R`

Replace the R `optimal_parameter()` body to call `optimal_parameter_cpp()`.

- [ ] **Step 1: Write integration test**

Append to `tests/testthat/test_reml_cpp.R`:
```r
test_that("Full ebp() with C++ REML produces same results as before", {
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
```

- [ ] **Step 2: Modify R/optimal_parameter.R**

Replace the body of `optimal_parameter()` to call C++. The key challenge is that the R function receives unsorted smp_data, but the C++ function needs data sorted by domain with contiguous domain_ids. The R wrapper handles this.

New `optimal_parameter()`:
```r
optimal_parameter <- function(generic_opt,
                              fixed,
                              smp_data,
                              smp_domains,
                              transformation,
                              interval,
                              control) {
  if (transformation != "no" &&
    transformation != "log") {

    if (transformation == "box.cox" && any(interval == "default")) {
      interval <- c(-1, 2)
    } else if (transformation == "dual" && any(interval == "default")) {
      interval <- c(0, 2)
    } else if (transformation == "log.shift" && any(interval == "default")) {
      span <- range(smp_data[paste(fixed[2])])
      if ((span[1] + 1) <= 1) {
        lower <- abs(span[1]) + 1
      } else {
        lower <- 0
      }
      upper <- diff(span) / 2
      interval <- c(lower, upper)
    }

    # Sort data by domain for C++ (requires contiguous domain blocks)
    smp_data_sorted <- smp_data[order(smp_data[[smp_domains]]), ]
    y <- as.numeric(smp_data_sorted[[as.character(fixed[[2]])]])
    X <- model.matrix(fixed, smp_data_sorted)
    domain_factor <- as.factor(smp_data_sorted[[smp_domains]])
    domain_ids <- as.integer(domain_factor)
    n_d <- as.integer(table(domain_factor))

    optimal_parameter <- optimal_parameter_cpp(
      y = y, X = X,
      domain_ids = domain_ids, n_d = n_d,
      transformation = transformation,
      lower = interval[1], upper = interval[2]
    )
  } else {
    optimal_parameter <- NULL
  }

  return(optimal_parameter)
}
```

Keep `generic_opt` and `reml` functions in the file unchanged — they may still be useful for debugging or as fallback.

- [ ] **Step 3: Compile and run tests**

```bash
Rscript -e 'Rcpp::compileAttributes(".")'
Rscript -e 'pkgbuild::compile_dll(".")'
Rscript -e 'devtools::test(filter = "reml_cpp")'
```

Expected: All REML tests pass, including integration tests.

- [ ] **Step 4: Run full test suite**

```bash
Rscript -e 'devtools::test()'
```

Expected: ALL tests pass. The existing `test_point_estimation.R` benchmark tests should still match because the optimal lambda should be the same (within tolerance).

- [ ] **Step 5: If benchmark tests fail, debug**

The most likely cause of failure is data ordering. The C++ function expects data sorted by domain. If `point_estim()` passes `framework$smp_data` which is already sorted (it is — `framework_ebp()` sorts at line 71), this should work. But if not, check the ordering.

- [ ] **Step 6: Commit**

```bash
git add R/optimal_parameter.R tests/testthat/test_reml_cpp.R
git commit -m "feat: wire optimal_parameter() to C++ REML optimization"
```

---

### Task 5: Benchmark and Final Verification

**Files:**
- No new files

- [ ] **Step 1: Run the MSE benchmark**

```bash
cd "/home/lpv/Dropbox/Inglaterra_2019/PhD/Southampton/R Packages/emdi"
Rscript -e '
library(devtools)
load_all(".")
data("eusilcA_smp"); data("eusilcA_pop")

fixed <- eqIncome ~ gender + eqsize + cash + self_empl +
  unempl_ben + age_ben + surv_ben + sick_ben + dis_ben +
  rent + fam_allow + house_allow + cap_inv + tax_adj

cat("=== emdi (pure R) vs emdi2 (C++ REML) ===\n\n")

cat("--- L=50, MSE=TRUE, B=10 ---\n")
set.seed(42)
t_orig <- system.time(emdi::ebp(
  fixed = fixed, pop_data = eusilcA_pop, pop_domains = "district",
  smp_data = eusilcA_smp, smp_domains = "district",
  L = 50, MSE = TRUE, B = 10))
cat("emdi  (pure R):", t_orig["elapsed"], "seconds\n")

set.seed(42)
t_cpp <- system.time(ebp(
  fixed = fixed, pop_data = eusilcA_pop, pop_domains = "district",
  smp_data = eusilcA_smp, smp_domains = "district",
  L = 50, MSE = TRUE, B = 10))
cat("emdi2 (C++):   ", t_cpp["elapsed"], "seconds\n")
cat("Speedup:       ", round(t_orig["elapsed"] / t_cpp["elapsed"], 1), "x\n")
'
```

- [ ] **Step 2: Run R CMD check**

```bash
Rscript -e 'devtools::check(args = c("--no-build-vignettes", "--no-manual"), build_args = c("--no-build-vignettes", "--no-manual"))' 2>&1 | tail -30
```

Expected: No new errors or warnings from our changes.

- [ ] **Step 3: Push to remote**

```bash
git push -u origin dev-emdi-reml-cpp
```

---

## Validation Checklist

The C++ implementation must match R within tolerance for:
- [ ] `std_transform_y_cpp` vs `box_cox_std`, `dual_std`, `log_shift_opt_std` (1e-10)
- [ ] `reml_loglik_cpp` vs `-logLik(lme())` (1e-4)
- [ ] `optimal_parameter_cpp` vs R `optimal_parameter()` (1e-4)
- [ ] Full `ebp()` output: indicators are valid
- [ ] Full `ebp(MSE=TRUE)`: MSE values are non-negative
- [ ] All existing tests continue to pass
