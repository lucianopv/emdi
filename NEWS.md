# emdi 2.2.0
* Extension of the ebp function to allow for population weights
* Extension of the ebp function to allow the aggregation of the estimates to different levels  
* a more flexible use of the custom_indicator agrument within the ebp function
* Bug fix in `optimal_parameter()` (transformations `box.cox`, `log.shift`, `dual`): the per-domain sample counts `n_d` are now built from `droplevels()` of the domain factor. If `smp_data[[smp_domains]]` carried unused factor levels (e.g. because the survey's `region2` was aligned to the census's level set for out-of-sample-domain coverage in `pop_data`), `table()` previously returned zero counts and the C++ sufficient-stats loop crashed with `Mat::rows(): indices out of bounds`. Added defensive `Rcpp::stop()` guards in `reml_loglik_cpp`, `lme_fit_cpp`, and `model_par_weighted_cpp` so a direct C++ call with ill-formed `n_d` fails fast with a clear message instead of an opaque Armadillo error.
* Performance fix in the C++ parametric bootstrap (`ebp(MSE = TRUE)` with `aggregate_to`): the aggregate-cell index cache was built with one `arma::find()` scan per cell, making construction `O(N_dom_agg * N_pop)`. This is a hard cliff for finer-than-model-domain output — at 3.7M population rows and ~140k grid cells it costs minutes before the first bootstrap iteration starts. It now uses the same two-pass counting sort as the Monte-Carlo path, which is `O(N_pop)` and independent of the number of aggregate cells. Results are unaffected: the returned MSE matrix is bit-for-bit `identical()` to the previous implementation for both coarsening and finer-output aggregations. Measured 1.115s to 0.100s at 25,000 cells on `eusilcA`. Regression test in `tests/testthat/test_agg_cache_perf.R`. Note that the equivalent construction in the Monte-Carlo path was already `O(N_pop)`; only the bootstrap path was affected.
* Performance fix in the jackknife MSE estimators for `fh()` (`mse_type = "jackknife"`, `"weighted_jackknife"`, and the measurement-error variant): the in-sample data frame and its `framework_FH()` were rebuilt inside the delete-one-domain loop although neither depends on the deleted domain, costing `m` redundant `model.matrix()` builds per fit. They are now built once, reducing `framework_FH()` calls per `fh()` from `1 + 2m` to `m + 2`. No numerical change — the stored jackknife benchmarks (`MSE_jack`, `MSE_wjack`) are unchanged. The wall-clock gain is modest (~15% at `m = 94`) because the loop also re-estimates `sigmau2` `m` times, which dominates.
* Improved progress reporting for the jackknife and spatial bootstrap MSE estimators. These previously printed one line per iteration reporting only the iteration number, so a long fit gave hundreds of lines of scrollback and no indication of how far along it was. Progress is now a single in-place line carrying position, percentage, elapsed time and projected finish time, preceded by a header naming the total and the absolute start time. Output is throttled by elapsed time rather than iteration count, and falls back to whole lines when `stderr` is not a terminal, so redirecting a long run to a log file no longer collapses it onto one line. New internal helpers in `R/progress.R`; tests in `tests/testthat/test_progress.R`.
* Bug fix in the C++ `Quintile_Share` kernel (unweighted `ebp()`): the 0.2/0.8 quantiles were computed with type-7 linear interpolation, while R's `qsr()` uses `wtd.quantile()` — the inverse-CDF ("step") rule — regardless of whether the weights are all one. The two rules select different cut points, so a different set of observations was averaged into the bottom and top quintile. On `eusilcA` (94 domains, unweighted) `Quintile_Share` differed from `emdi::ebp()` in 52 of 94 domains by up to 2.9% (`transformation = "no"`), 8.5% (`"log"`) and 4.3% (`"box.cox"`); bootstrap MSE for the indicator inherited the same error. Point estimates and MSE for all other indicators (`Mean`, `Head_Count`, `Poverty_Gap`, `Gini`, `Quantile_10`–`Quantile_90`), the weighted `Quintile_Share` path, and all `fh()` methods were unaffected. The kernel now uses the step rule, located by binary search so no cumulative-weight vector is allocated (no measurable runtime or memory cost). This also removes an inconsistency whereby unweighted `Quintile_Share` depended on whether `Gini` was among the requested indicators. Regression tests added in `tests/testthat/test_qsr_cpp.R`.

# emdi 2.1.3
* Improved summary with clearer notation of R2
* Updated Area Level (FH) vignette
* Minor improvements in checking T/F in if clauses
* Increased dependency to R (>= 4.2.0) corresponding to the imported package MuMin

# emdi 2.1.2
* Improved messages
* Copyrights updated
* Bug fix in direct variance estimation (direct)
* Bug fix in level orderings (ebp)
* Changed last name in description

# emdi 2.1.1
* Fix to account for the changed behavior of as.vector() on data frames

# emdi 2.1.0
* Extension of the ebp function to allow informative sampling
* Additional data-driven transformations for the ebp 
* New additional vignette
* Example in write.ods fixed

# emdi 2.0.3
* Robustifying tests to comply with alternative implementations
* Updated example for step


# emdi 2.0.2
* Many S3-methods in the style of stats and nlme are implemented for the classes direct, ebp and fh
* Structure of S3-classes has been cleaned up
* The bootstrap parameter in the fh-function has been changed from a single number to a single number or a numeric vector with two elements to allow for separately controlling the number of bootstrap iterations for the MSE estimation and the computation of the bootstrap based information criteria
* Renaming the robustness constant in the fh function
* Minor fixes in the documentation
* Reducing the sizes of the data sets used for test that tests to decrease testing time

# emdi 2.0.1
* Robustifying tests to comply with alternative BLAS/LAPACK implementations
* Robustifying tests to work with r-oldrel
* Updated R version dependency

# emdi 2.0.0
* Area-level models newly added via function `fh`
* All methods for `emdi model` are extended for `emdi model fh`
* Step function for area-level models newly added
* Three new data sets `eusilcA_smpAgg`, `eusilcA_popAgg` and 
`eusilcA_prox` have been integrated
* New additional vignette
* Change of argument order (model and direct) in function `compare_plot`
* Minor bug fixes in argument checkings and message handling
* Updated R version dependency

# emdi 1.1.7

* Minor typos corrected
* Unit-Tests adjusted to the forthcoming R-Version 4.0

# emdi 1.1.6

* New and updated references.

# emdi 1.1.5

* Tests updated to deal with new random number generation in R
* Some spelling improved

# emdi 1.1.4

* Fixed Bug in summary: R2 calculation with MuMIn is now fully working
* Added feature: formula used in fixed is now preserved, even if passed to ebp as a variable

# emdi 1.1.3

* The function `compare_plot` now benefits from a legend in all plots.
* Changes to be compatible with the forthcoming version of the MuMIn package.


# emdi 1.1.2

* The function `compare_plot` has been added to allow for an easy comparison 
between direct and model based estimates.
* Argument checks have been added, and improved
* Additional example in `map_plot` in order to explain the mapping table
* The datasets have been improved to allow for more realistic examples
* Small bug fixes
* Updated Vignette

# emdi 1.1.1

* The function `ebp` benefits from a new parameter called `seed` that allows 
reproducibility even when the function is run in parallel mode.
* Argument checks have been added, and improved
* Additional example in `map_plot` and `direct` 
* Updated Vignette

# emdi 1.1.0

* A new function `direct` is made available, which provides direct estimation for small areas.
* The function `ebp` now allows for a user-defined threshold.
* The function `ebp` is now able to perform a semi-parametric wild bootstrap for MSE estimation.
* The function `ebp` has new default value for parallelization that automatically adopts for the operating system.
* The two data sets `eusilcA_smp` and `eusilcA_pop` have been updated.
* For the function `map_plot` additional customization is now applicable.
* All methods for `emdi model` except plot are extended for `emdi direct`.
* `subset` and `as.data.frame` have been added as methods for class `emdi.estimators`