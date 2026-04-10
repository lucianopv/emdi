# C++ LME Fit Replacing nlme::lme() in Bootstrap

## Problem

Each MSE bootstrap iteration calls `nlme::lme()` to fit the random-intercept model on the bootstrap sample. While `lme()` itself is ~7% of iteration time on the small eusilcA dataset, on larger datasets (e.g., Mozambique with 11K survey obs) the overhead is significant. Combined with the already-implemented C++ REML optimization for `optimal_parameter`, replacing `lme()` with a direct C++ fit eliminates all R-level mixed model fitting from the bootstrap loop.

## Solution

Create `lme_fit_cpp()` — a C++ function that fits the random-intercept REML model and returns all parameters that `model_par()` currently extracts from the nlme object:
- `betas`: fixed effect coefficients
- `sigma2_e`: residual variance
- `sigma2_u`: random effect variance
- `rand_eff`: BLUPs of domain random effects

For the weighted case, create `model_par_weighted_cpp()` that takes sigma2_e/sigma2_u from the fit and re-computes betas and random effects using the pseudo-EB approach with survey weights.

## Key insight: bootstrap vs initial call

- **Initial `ebp()` call**: Still uses `nlme::lme()` to produce a proper model object for downstream methods (`summary`, `plot`, `residuals`, etc.)
- **Bootstrap iterations** (inside `mse_estim()`): Only need the extracted parameters — the model object is never used. This is where we use C++.

The `point_estim()` function gets a new parameter `fast_fit = FALSE`. When `TRUE`, it skips `nlme::lme()` and uses `lme_fit_cpp()` + builds the parameter list directly. The bootstrap path in `mse_estim()` passes `fast_fit = TRUE`.

## BLUP computation

For domain i with n_i observations:
```
gamma_i = sigma2_u / (sigma2_u + sigma2_e / n_i)
u_hat_i = gamma_i * (y_bar_i - x_bar_i' * beta)
```
where y_bar_i = mean(y in domain i), x_bar_i = mean(X in domain i).

## Weighted case

Uses sigma2_e, sigma2_u from the C++ fit, then computes:
- `delta2[d] = sum(w_d^2) / sum(w_d)^2`
- `gamma_weight[d] = sigma2_u / (sigma2_u + sigma2_e * delta2[d])`
- Weighted betas via `solve(den) %*% num` (iterative reweighting per domain)
- Weighted random effects: `gamma_weight[d] * (mean_dep[d] - mean_indep[d,] %*% betas)`

## Files

| File | Action | Responsibility |
|------|--------|---------------|
| `src/lme_fit.cpp` | Create | lme_fit_cpp, model_par_weighted_cpp |
| `R/point_estimation.R` | Modify | point_estim() accepts fast_fit parameter, uses C++ when TRUE |
| `R/mse_estimation.R` | Modify | mse_estim() passes fast_fit=TRUE to point_estim() |
| `tests/testthat/test_lme_fit_cpp.R` | Create | Verify C++ fit matches nlme::lme() |

## Validation

The C++ fit must match nlme::lme() within floating-point tolerance for:
- betas (1e-8)
- sigma2_e, sigma2_u (1e-6)
- random effects / BLUPs (1e-6)
- Full ebp() output: indicators remain valid
- MSE bootstrap: produces equivalent results
