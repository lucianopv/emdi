# Full C++ Parametric Bootstrap Loop

## Problem

Even with individual C++ functions for MC simulation, REML optimization, superpopulation, and bootstrap sample generation, each bootstrap iteration still crosses the R↔C++ boundary ~10 times and spends ~39% of iteration time on R glue code (data manipulation, function dispatch, object creation). Over B=200 iterations this adds up significantly.

## Solution

Create a single `parametric_bootstrap_cpp()` that runs the **entire** B-iteration parametric bootstrap loop in C++, returning only the final MSE matrix to R. This eliminates all per-iteration R overhead.

### Scope

**Included:**
- Parametric bootstrap (not wild)
- All transformation types (no, log, box.cox, dual, log.shift)
- Unweighted and weighted cases
- selected_domains (handled automatically via pre-filtered framework vectors)
- aggregate_to (pass aggregated domain IDs)
- Standard 10 indicators computed in C++

**Excluded (R fallback):**
- Wild bootstrap — stays in existing R loop
- Custom indicators — when present, C++ returns y_mcmc per iteration so R computes them
- Parallel execution — the C++ loop is single-threaded (R parallelMap handles parallelism at a higher level if needed later)

### Decision logic in R

```r
parametric_bootstrap <- function(...) {
  if (boot_type == "parametric" && is.null(custom_indicator)) {
    # Full C++ path
    parametric_bootstrap_cpp(...)
  } else {
    # Existing R loop (wild bootstrap, or parametric with custom indicators)
    ...existing code...
  }
}
```

## Architecture

### What happens inside each C++ iteration

```
for b in 1..B:
  1. gen_superpop       — generate superpopulation (eps + vu + back-transform)
  2. true_indicators    — compute indicators on superpopulation per domain
  3. gen_bootstrap_smp  — generate bootstrap sample Y_smp_b
  4. optimal_parameter  — find optimal lambda (Brent over lambda → Brent over theta → REML)
                          [skipped for "no" and "log" transformations]
  5. data_transform     — transform bootstrap sample y with optimal lambda
  6. lme_fit            — fit random-intercept REML model, get betas/sigma2_e/sigma2_u/BLUPs
  7. model_par          — extract/compute model parameters (unweighted or weighted)
  8. gen_model          — compute sigmav2, mu, mu_fixed for generating model
  9. monte_carlo        — L iterations of MC simulation with indicators
  10. mse_accum         — accumulate (bootstrap_estimates - true_indicators)^2
```

Return: MSE matrix [N_dom x n_indicators] = accumulated / B

### New C++ functions needed

All go in `src/lme_fit.cpp`:

**1. `data_transform_cpp(y, transformation, lambda)`**
Non-standardized forward transformation (with shift). Returns transformed y and shift parameter. Matches `data_transformation()` but operates on a vector, not a data frame.

**2. `lme_fit_cpp(y_transformed, X, n_d, D)`**
Fit random-intercept REML model. Returns betas, sigma2_e, sigma2_u, random effects (BLUPs), gamma values. Reuses the Brent/REML machinery from reml_optimization.cpp.

**3. `model_par_weighted_cpp(y_transformed, X, weights, n_d, D, sigma2_e, sigma2_u)`**
Weighted pseudo-EB parameter estimation. Takes sigma2_e/sigma2_u from lme_fit, computes weighted betas, delta2, gamma_weight, weighted random effects.

**4. `parametric_bootstrap_cpp(...)`**
The main exported function. Takes all framework data as vectors/matrices. Runs the B-iteration loop internally. Returns MSE matrix.

### Signature for parametric_bootstrap_cpp

```cpp
// [[Rcpp::export]]
Rcpp::List parametric_bootstrap_cpp(
    // Population data
    const arma::mat& X_pop,          // population design matrix [N_pop x p]
    const arma::vec& mu_fixed_orig,  // X_pop %*% original betas [N_pop]
    const arma::vec& rand_eff_orig,  // original random effects [N_dom_pop]
    const arma::ivec& n_pop,         // pop count per domain [N_dom_pop]
    const arma::ivec& pop_domain_ids,// domain ID per pop obs [N_pop]
    const arma::ivec& obs_dom,       // is pop obs in-sample domain? [N_pop]
    const arma::ivec& dist_obs_dom,  // is domain in-sample? [N_dom_pop]
    const arma::vec& pop_weights,    // population weights [N_pop]
    // Sample data  
    const arma::mat& X_smp,          // sample design matrix [N_smp x p]
    const arma::vec& y_smp_orig,     // original sample y (untransformed) [N_smp]
    const arma::ivec& n_smp,         // sample count per domain [N_dom_smp]
    const arma::ivec& smp_to_pop_map,// maps smp domain idx to pop domain idx [N_dom_smp]
    // Model parameters (from original fit)
    double sigmae2_orig,
    double sigmau2_orig,
    const arma::vec& sigmav2_orig,   // [N_dom_smp_selected]
    const arma::vec& mu_orig,        // constant part for pop [N_pop]
    const arma::vec& mu_fixed_pop,   // X_pop %*% betas [N_pop]
    // Dimensions
    int N_pop, int N_smp, int N_dom_pop, int N_dom_smp,
    int N_dom_smp_selected, int N_dom_unobs,
    // Algorithm parameters
    int B, int L,
    double threshold,
    const std::string& transformation,
    double lambda_orig,              // original optimal lambda
    double shift_orig,               // original shift
    const arma::vec& interval,       // optimization interval for lambda [2]
    // Optional aggregate
    Rcpp::Nullable<Rcpp::IntegerVector> agg_domain_ids = R_NilValue,
    int N_dom_agg = 0,
    // Weights (NULL if unweighted)
    Rcpp::Nullable<Rcpp::NumericVector> smp_weights = R_NilValue
);
// Returns List with:
//   "mse": matrix [N_dom x 10] mean squared errors
//   "true_indicators_all": (optional, for debugging)
```

## R Integration

Modify `parametric_bootstrap()` in `R/mse_estimation.R`:
- Check conditions for C++ path (parametric, no custom indicators)
- Extract all needed vectors from the framework object
- Call `parametric_bootstrap_cpp()`
- Format result as data.frame matching existing output

The existing R loop remains as fallback for wild bootstrap and custom indicators.

## Validation

1. Compare MSE output from C++ loop vs R loop on eusilcA data (same seed)
2. Compare on all transformation types
3. Verify with aggregate_to
4. Verify weighted case
5. Full ebp(MSE=TRUE) integration test
6. Benchmark on eusilcA and Mozambique data
