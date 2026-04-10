# C++ REML Optimization for MSE Bootstrap Speedup

## Problem

The MSE bootstrap in emdi2 calls `point_estim()` B times (typically B=200-500). Each call runs `optimal_parameter()` which uses R's `optimize()` to find the best transformation parameter lambda by minimizing the negative REML log-likelihood. Each evaluation calls `nlme::lme()` to fit a full mixed model. With ~9 evaluations per `optimize()` call, that's ~9 lme() fits per bootstrap iteration.

**Profiling shows `optimal_parameter()` accounts for 72% of each bootstrap iteration's time.** The actual model fit (`lme()`) for the final model is only 7%. The Monte Carlo simulation (already in C++) is 19%.

This is methodologically required: Rojas-Perilla et al. (2019) explicitly specify that lambda must be re-estimated in each bootstrap iteration to capture the additional uncertainty from estimating the transformation parameter.

## Solution

Replace the `reml()` function (which calls `lme()`) with a C++ function `reml_loglik_cpp()` that computes the REML log-likelihood directly. R's `optimize()` calls this C++ function instead of the R/lme chain.

The nested error model `y = Xb + Zu + e` with random intercept per domain has a **closed-form REML log-likelihood** that can be evaluated using per-domain sufficient statistics. No iterative model fitting is needed — we profile over a single parameter `theta = sigma2_u / sigma2_e` and compute everything analytically.

### Verified approach

Manual computation of the profile REML log-likelihood (optimizing over theta via R's `optimize()`) matches `nlme::lme()` output to 10 decimal places.

## Architecture

```
R optimize() over lambda
  └── generic_opt_cpp(lambda)           [new R wrapper]
        ├── std_transform_cpp(y, lambda) [new C++ function]
        ├── optimize() over log(theta)   [R, ~15 evaluations]
        │     └── reml_profile_cpp(...)  [new C++ function]
        │           ├── Per-domain XtVinvX, XtVinvy accumulation
        │           ├── Solve for beta_hat
        │           ├── Profile sigma2_e
        │           └── Return -REML loglik
        └── Return -REML loglik at optimal theta
```

The outer `optimize()` over lambda stays in R (~9 evaluations). Each evaluation calls a C++ function that:
1. Applies the standardized transformation to y
2. Runs an inner `optimize()` over log(theta) where each evaluation is a fast C++ REML profile likelihood computation
3. Returns the negative REML log-likelihood

### Alternative: Single C++ function for the whole thing

Instead of two nested R `optimize()` calls, we could have a single C++ function `optimal_parameter_cpp(y, X, domain_ids, n_d, transformation, interval)` that does both the lambda and theta optimization internally using a C++ optimizer (e.g., Brent's method from Armadillo or a simple golden section search). This avoids R↔C++ overhead on the inner loop.

**Recommended: Single C++ function.** The inner theta optimization is called ~9×15 = 135 times per bootstrap iteration. Keeping it entirely in C++ avoids significant R overhead.

## C++ Functions

### 1. `std_transform_y_cpp(y, transformation, lambda)` 
Standardized forward transformation (Box-Cox, Dual, Log-shift with geometric mean scaling).
- Input: raw y vector, transformation type, lambda
- Output: transformed y vector
- Must match `box_cox_std()`, `dual_std()`, `log_shift_opt_std()` exactly

### 2. `reml_profile_cpp(log_theta, y_std, X, domain_starts, domain_sizes, n, p, D)`
Evaluate profile REML negative log-likelihood at a given theta = exp(log_theta).
- Uses per-domain sufficient statistics: X'V^{-1}X, X'V^{-1}y, r'V^{-1}r
- V_d^{-1}/sigma2_e = I - c_d * J where c_d = theta/(1 + n_d * theta)
- Profiles sigma2_e = (r'V^{-1}r / sigma2_e) / (n - p)
- Returns -REML loglik

### 3. `reml_loglik_cpp(lambda, y, X, domain_ids, n_d, transformation, n, p, D)`
Full REML evaluation at a given lambda:
1. Apply `std_transform_y_cpp(y, transformation, lambda)` to get y_std
2. Optimize over log(theta) using Brent's method to find optimal variance ratio
3. Return -REML loglik at optimal (lambda, theta)

### 4. `optimal_parameter_cpp(y, X, domain_ids, n_d, transformation, interval, n, p, D)`
Find optimal lambda by minimizing `reml_loglik_cpp` over the given interval using Brent's method.
Returns: optimal lambda value.

This single exported function replaces the entire `optimal_parameter()` → `generic_opt()` → `reml()` → `lme()` chain.

## REML Mathematics

For the nested error model with D domains of sizes n_1, ..., n_D:

**Block-diagonal covariance**: V = diag(V_1, ..., V_D) where V_d = sigma2_e * I_{n_d} + sigma2_u * J_{n_d}

**Inverse** (Woodbury): V_d^{-1} = (1/sigma2_e)(I - c_d * J) where c_d = sigma2_u / (sigma2_e + n_d * sigma2_u)

With theta = sigma2_u / sigma2_e: c_d = theta / (1 + n_d * theta)

**Per-domain sufficient statistics** (all O(n_d * p) per domain):
- S_xx_d = X_d' X_d  (p x p)
- S_xy_d = X_d' y_d  (p x 1) 
- S_yy_d = y_d' y_d  (scalar)
- x_bar_d = colSums(X_d)  (p x 1, column sums)
- y_bar_d = sum(y_d)  (scalar)

**Accumulated statistics** (given theta):
- X'V^{-1}X / sigma2_e = sum_d [ S_xx_d - c_d * x_bar_d * x_bar_d' ]
- X'V^{-1}y / sigma2_e = sum_d [ S_xy_d - c_d * x_bar_d * y_bar_d ]

**Beta**: beta_hat = (X'V^{-1}X)^{-1} X'V^{-1}y

**Profiled sigma2_e**: sigma2_e_hat = quad / (n - p) where quad = r'V^{-1}r / sigma2_e

**Log-determinants**:
- log|V| = sum_d [ (n_d - 1)*log(sigma2_e) + log(sigma2_e + n_d*sigma2_u) ]
- log|X'V^{-1}X| computed via Cholesky

**REML log-likelihood**:
-2 * l_REML = (n-p)*log(2pi) + log|V| + log|X'V^{-1}X| + r'V^{-1}r

## R Integration

Modify `optimal_parameter()` to call `optimal_parameter_cpp()` when available:

```r
optimal_parameter <- function(generic_opt, fixed, smp_data, smp_domains,
                              transformation, interval, control) {
  if (transformation != "no" && transformation != "log") {
    # ... interval setup (unchanged) ...
    
    y <- as.numeric(smp_data[[as.character(fixed[[2]])]])
    X <- model.matrix(fixed, smp_data)
    domain_ids <- as.integer(as.factor(smp_data[[smp_domains]]))
    n_d <- as.integer(table(as.factor(smp_data[[smp_domains]])))
    
    optimal_parameter <- optimal_parameter_cpp(
      y = y, X = X, domain_ids = domain_ids, n_d = n_d,
      transformation = transformation,
      lower = interval[1], upper = interval[2]
    )
  } else {
    optimal_parameter <- NULL
  }
  return(optimal_parameter)
}
```

## Validation

The C++ implementation must produce **identical results** (within floating-point tolerance ~1e-10) to the current R implementation for:
1. The REML log-likelihood at any given (lambda, theta)
2. The optimal theta at any given lambda
3. The optimal lambda returned by optimal_parameter
4. The full ebp() output (point estimates and MSE)

Test against all 3 transformation types: box.cox, dual, log.shift.

## Expected Impact

Current per-bootstrap-iteration: ~0.77s (of which optimal_parameter = 0.35s = 72%)

The C++ REML evaluation eliminates ~9 lme() calls. Each lme() call involves:
- R object creation/manipulation overhead
- Generic optimizer setup in R
- Full mixed model machinery (much more general than needed)

The C++ version does only the minimal math needed. Conservative estimate: 10-50x faster per REML evaluation, making optimal_parameter drop from 0.35s to ~0.01-0.03s.

**Expected overall MSE speedup: 2-3x on top of existing C++ gains** (from ~15s to ~5-7s for the benchmark case).

## References

- Rojas-Perilla, N., Pannier, S., Schmid, T., & Tzavidis, N. (2019). Data-driven transformations in small area estimation. *Journal of the Royal Statistical Society Series A*, 182(1), 121-148.
- Molina, I., & Rao, J. N. K. (2010). Small area estimation of poverty indicators. *Canadian Journal of Statistics*, 38(3), 369-385.
