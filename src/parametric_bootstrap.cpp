#include <RcppArmadillo.h>
#include <chrono>
#include <ctime>
#include "progress.h"
// [[Rcpp::depends(RcppArmadillo)]]

#ifdef _OPENMP
  #include <omp.h>
#endif

// Forward declarations for functions defined in other translation units.

// From transformations.cpp (arma version for internal use, no copy)
arma::vec back_transform_arma(const arma::vec& y,
                               const std::string& transformation,
                               double lambda, double shift);
// From transformations.cpp (Rcpp version for R-facing API)
Rcpp::NumericVector back_transform_cpp(const arma::vec& y,
                                       const std::string& transformation,
                                       double lambda, double shift);

// From indicators.cpp
arma::vec compute_domain_indicators_cpp(const arma::vec& y,
                                        const arma::vec& weights,
                                        double threshold);
arma::vec compute_domain_indicators_selective_cpp(const arma::vec& y,
                                                    const arma::vec& weights,
                                                    double threshold,
                                                    int indicator_mask);

// From reml_optimization.cpp
double optimal_parameter_cpp(const arma::vec& y, const arma::mat& X,
                              const arma::ivec& domain_ids,
                              const arma::ivec& n_d,
                              const std::string& transformation,
                              double lower, double upper);

// From lme_fit.cpp
Rcpp::List data_transform_cpp(const arma::vec& y_raw,
                               const std::string& transformation,
                               double lambda);
Rcpp::List lme_fit_cpp(const arma::vec& y_transformed,
                        const arma::mat& X, const arma::ivec& n_d);
Rcpp::List model_par_weighted_cpp(const arma::vec& y_transformed,
                                   const arma::mat& X,
                                   const arma::vec& weights,
                                   const arma::ivec& n_d,
                                   double sigma2_e, double sigma2_u);

// ---------------------------------------------------------------------------
// parametric_bootstrap_cpp
//
// Runs the full B-iteration parametric bootstrap for MSE estimation.
// Each iteration: generate superpopulation, compute true indicators,
// generate bootstrap sample, fit model, run MC simulation, accumulate MSE.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat parametric_bootstrap_cpp(
    const arma::mat& X_pop,
    const arma::vec& mu_fixed_orig,
    const arma::ivec& n_pop,
    const arma::ivec& obs_dom,
    const arma::ivec& dist_obs_dom,
    const arma::vec& pop_weights,
    int N_pop, int N_dom_pop,
    const arma::mat& X_smp,
    const arma::ivec& n_smp,
    const arma::ivec& smp_domain_ids,
    const arma::ivec& smp_to_pop_map,
    int N_smp, int N_dom_smp,
    const arma::vec& betas_orig,
    double sigmae2_orig,
    double sigmau2_orig,
    int N_dom_smp_selected, int N_dom_unobs,
    int B, int L,
    double threshold,
    const std::string& transformation,
    double lambda_orig,
    double shift_orig,
    double interval_lower, double interval_upper,
    Rcpp::Nullable<Rcpp::IntegerVector> agg_domain_ids_pop = R_NilValue,
    int N_dom_agg = 0,
    Rcpp::Nullable<Rcpp::NumericVector> smp_weights = R_NilValue,
    int indicator_mask = 0x3FF,
    int threads = 1
) {
  if (threads < 1) threads = 1;

  const int n_indicators = 10;

  // Determine aggregation
  bool use_agg = agg_domain_ids_pop.isNotNull() && N_dom_agg > 0;
  arma::ivec agg_ids;
  int N_dom_ind; // number of domains for indicator output
  if (use_agg) {
    agg_ids = Rcpp::as<arma::ivec>(agg_domain_ids_pop.get());
    N_dom_ind = N_dom_agg;
  } else {
    N_dom_ind = N_dom_pop;
  }

  // Pre-build index vectors for aggregated domains.
  // Two-pass counting sort: O(N_pop) total, independent of N_dom_ind. A
  // per-domain arma::find() would be O(N_dom_ind * N_pop), which is a hard
  // cliff for finer-than-model-domain output (many small aggregate cells).
  // Mirrors the construction in monte_carlo.cpp.
  std::vector<arma::uvec> agg_idx_cache;
  if (use_agg) {
    agg_idx_cache.resize(N_dom_ind);
    int N_pop_agg = (int)agg_ids.n_elem;
    std::vector<int> bucket_count(N_dom_ind, 0);
    for (int i = 0; i < N_pop_agg; i++) {
      bucket_count[agg_ids(i) - 1]++;
    }
    for (int d = 0; d < N_dom_ind; d++) {
      agg_idx_cache[d].set_size(bucket_count[d]);
    }
    std::vector<int> fill_pos(N_dom_ind, 0);
    for (int i = 0; i < N_pop_agg; i++) {
      int d = agg_ids(i) - 1;
      agg_idx_cache[d](fill_pos[d]) = i;
      fill_pos[d]++;
    }
  }

  // Determine if weighted
  bool use_weights = smp_weights.isNotNull();
  arma::vec smp_wts;
  if (use_weights) {
    smp_wts = Rcpp::as<arma::vec>(smp_weights.get());
  }

  // Precompute sqrt of original variances
  double sd_e_orig      = std::sqrt(sigmae2_orig);
  double sd_e_unobs_orig = std::sqrt(sigmae2_orig + sigmau2_orig);
  double sd_u_orig      = std::sqrt(sigmau2_orig);

  // Build a map from population domain index to sample domain index
  // smp_to_pop_map[s] = pop domain index (1-based) for sample domain s
  // We need reverse: for each pop domain d (0-based), which sample domain s maps to it?
  // pop_to_smp_map[d] = sample domain index (0-based), or -1 if not found
  std::vector<int> pop_to_smp_map(N_dom_pop, -1);
  for (int s = 0; s < N_dom_smp; s++) {
    int pop_idx = smp_to_pop_map[s]; // 1-based or NA
    if (pop_idx != NA_INTEGER && pop_idx >= 1 && pop_idx <= N_dom_pop) {
      pop_to_smp_map[pop_idx - 1] = s;
    }
  }

  // Precompute domain offsets for population (contiguous blocks)
  std::vector<int> pop_dom_offset(N_dom_pop);
  {
    int off = 0;
    for (int d = 0; d < N_dom_pop; d++) {
      pop_dom_offset[d] = off;
      off += n_pop[d];
    }
  }

  // Precompute domain offsets for sample (contiguous blocks)
  std::vector<int> smp_dom_offset(N_dom_smp);
  {
    int off = 0;
    for (int s = 0; s < N_dom_smp; s++) {
      smp_dom_offset[s] = off;
      off += n_smp[s];
    }
  }

  // MSE accumulator [N_dom_ind x n_indicators]
  arma::mat mse_accum(N_dom_ind, n_indicators, arma::fill::zeros);

  // Whether lambda optimization is needed
  bool needs_lambda_opt = (transformation != "no" && transformation != "log");

  // =========================================================================
  // Main bootstrap loop
  // =========================================================================
  // steady_clock for elapsed time (monotonic, sub-second); wall-clock epoch
  // only for projecting the finish time onto the user's clock.
  const auto t0 = std::chrono::steady_clock::now();
  const double start_epoch = (double)std::time(NULL);
  double last_report = 0.0;
  Rprintf("%s\n", emdi::progress_header("Bootstrap MSE", B,
                                        "bootstrap iteration", start_epoch).c_str());

  for (int b = 0; b < B; b++) {

    Rcpp::checkUserInterrupt();

    // =====================================================================
    // Step 1: Generate superpopulation
    // =====================================================================

    arma::vec eps_superpop(N_pop);

    // RNG step 1: eps for in-sample obs
    for (int i = 0; i < N_pop; i++) {
      if (obs_dom[i] != 0) {
        eps_superpop[i] = R::rnorm(0.0, sd_e_orig);
      }
    }
    // RNG step 2: eps for out-of-sample obs
    for (int i = 0; i < N_pop; i++) {
      if (obs_dom[i] == 0) {
        eps_superpop[i] = R::rnorm(0.0, sd_e_unobs_orig);
      }
    }

    // RNG step 3: domain random effects
    arma::vec vu_tmp(N_dom_pop);
    for (int d = 0; d < N_dom_pop; d++) {
      vu_tmp[d] = R::rnorm(0.0, sd_u_orig);
    }

    // Expand vu_tmp to population level
    arma::vec vu_pop(N_pop);
    {
      int pos = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        int nd = n_pop[d];
        for (int j = 0; j < nd; j++) {
          vu_pop[pos++] = vu_tmp[d];
        }
      }
    }

    // Y on transformed scale
    arma::vec Y_pop_notrans = mu_fixed_orig + eps_superpop + vu_pop;

    // Back-transform (arma version — no Rcpp copy overhead)
    arma::vec Y_pop_b = back_transform_arma(Y_pop_notrans,
                                             transformation,
                                             lambda_orig, shift_orig);

    // Replace non-finite with 0
    for (int i = 0; i < N_pop; i++) {
      if (!std::isfinite(Y_pop_b[i])) Y_pop_b[i] = 0.0;
    }

    // =====================================================================
    // Step 2: Compute true indicators on superpopulation
    // =====================================================================

    arma::mat true_indicators(N_dom_ind, n_indicators);

    if (use_agg) {
      for (int d = 0; d < N_dom_ind; d++) {
        const arma::uvec& idx = agg_idx_cache[d];
        arma::vec y_d = Y_pop_b.elem(idx);
        arma::vec w_d = pop_weights.elem(idx);
        true_indicators.row(d) = compute_domain_indicators_selective_cpp(y_d, w_d, threshold, indicator_mask).t();
      }
    } else {
      int offset = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        int nd = n_pop[d];
        arma::vec y_d = Y_pop_b.subvec(offset, offset + nd - 1);
        arma::vec w_d = pop_weights.subvec(offset, offset + nd - 1);
        true_indicators.row(d) = compute_domain_indicators_selective_cpp(y_d, w_d, threshold, indicator_mask).t();
        offset += nd;
      }
    }

    // =====================================================================
    // Step 3: Generate bootstrap sample
    // =====================================================================

    // RNG step 4: eps for sample
    arma::vec eps_smp(N_smp);
    for (int i = 0; i < N_smp; i++) {
      eps_smp[i] = R::rnorm(0.0, sd_e_orig);
    }

    // RNG step 5: vu for sample domains (use vu_tmp via smp_to_pop_map)
    arma::vec vu_for_smp(N_dom_smp);
    for (int s = 0; s < N_dom_smp; s++) {
      int pop_idx = smp_to_pop_map[s]; // 1-based or NA
      if (pop_idx != NA_INTEGER && pop_idx >= 1 && pop_idx <= N_dom_pop) {
        vu_for_smp[s] = vu_tmp[pop_idx - 1];
      } else {
        vu_for_smp[s] = R::rnorm(0.0, sd_u_orig);
      }
    }

    // Expand vu_for_smp to sample level
    arma::vec vu_smp(N_smp);
    {
      int pos = 0;
      for (int s = 0; s < N_dom_smp; s++) {
        int nd = n_smp[s];
        for (int j = 0; j < nd; j++) {
          vu_smp[pos++] = vu_for_smp[s];
        }
      }
    }

    // Y_smp on transformed scale
    arma::vec mu_smp = X_smp * betas_orig;
    arma::vec Y_smp_notrans = mu_smp + eps_smp + vu_smp;

    // Back-transform (arma version — no Rcpp copy overhead)
    arma::vec Y_smp_b = back_transform_arma(Y_smp_notrans,
                                             transformation,
                                             lambda_orig, shift_orig);

    // Replace non-finite with 0
    for (int i = 0; i < N_smp; i++) {
      if (!std::isfinite(Y_smp_b[i])) Y_smp_b[i] = 0.0;
    }

    // =====================================================================
    // Step 4: Find optimal lambda on bootstrap sample (if needed)
    // =====================================================================

    double lambda_b = lambda_orig;
    if (needs_lambda_opt) {
      lambda_b = optimal_parameter_cpp(Y_smp_b, X_smp, smp_domain_ids,
                                        n_smp, transformation,
                                        interval_lower, interval_upper);
    }

    // =====================================================================
    // Step 5: Transform bootstrap sample
    // =====================================================================

    Rcpp::List dt_result = data_transform_cpp(Y_smp_b, transformation, lambda_b);
    arma::vec y_trans_b = Rcpp::as<arma::vec>(dt_result["y"]);
    double shift_b = 0.0;
    if (!Rf_isNull(dt_result["shift"])) {
      shift_b = Rcpp::as<double>(dt_result["shift"]);
    }

    // =====================================================================
    // Step 6: Fit LME on transformed bootstrap sample
    // =====================================================================

    Rcpp::List lme_result = lme_fit_cpp(y_trans_b, X_smp, n_smp);
    arma::vec betas_b = Rcpp::as<arma::vec>(lme_result["betas"]);
    double sigma2_e_b = Rcpp::as<double>(lme_result["sigma2_e"]);
    double sigma2_u_b = Rcpp::as<double>(lme_result["sigma2_u"]);
    arma::vec rand_eff_b = Rcpp::as<arma::vec>(lme_result["rand_eff"]);
    arma::vec gamma_b = Rcpp::as<arma::vec>(lme_result["gamma"]);

    // =====================================================================
    // Step 7: model_par (weighted case)
    // =====================================================================

    arma::vec gammaw_b;
    if (use_weights) {
      Rcpp::List wp_result = model_par_weighted_cpp(y_trans_b, X_smp,
                                                     smp_wts, n_smp,
                                                     sigma2_e_b, sigma2_u_b);
      betas_b = Rcpp::as<arma::vec>(wp_result["betas"]);
      rand_eff_b = Rcpp::as<arma::vec>(wp_result["rand_eff"]);
      gammaw_b = Rcpp::as<arma::vec>(wp_result["gammaw"]);
      // gamma_b stays from lme_fit for unweighted sigmav2 calc
    }

    // =====================================================================
    // Step 8: gen_model computation
    // =====================================================================

    // Compute sigmav2 for in-sample domains
    // For unweighted: gamma_d = sigma2_u_b / (sigma2_u_b + sigma2_e_b / n_smp_d)
    // For weighted: use gammaw from model_par_weighted
    // sigmav2_d = sigma2_u_b * (1 - gamma_d)
    arma::vec sigmav2_all(N_dom_smp);
    if (use_weights) {
      for (int s = 0; s < N_dom_smp; s++) {
        sigmav2_all[s] = sigma2_u_b * (1.0 - gammaw_b[s]);
      }
    } else {
      for (int s = 0; s < N_dom_smp; s++) {
        double gam = sigma2_u_b / (sigma2_u_b + sigma2_e_b / (double)n_smp[s]);
        sigmav2_all[s] = sigma2_u_b * (1.0 - gam);
      }
    }

    // Map sigmav2 from sample domains to population domains (in-sample only)
    // sigmav2_selected: one entry per in-sample pop domain (in order of dist_obs_dom)
    arma::vec sigmav2_selected(N_dom_smp_selected);
    {
      int sel_idx = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        if (dist_obs_dom[d]) {
          int s = pop_to_smp_map[d];
          if (s >= 0 && s < N_dom_smp) {
            sigmav2_selected[sel_idx] = sigmav2_all[s];
          } else {
            sigmav2_selected[sel_idx] = sigma2_u_b; // fallback
          }
          sel_idx++;
        }
      }
    }

    // Build rand_eff for population domains
    // rand_eff_b has N_dom_smp entries; map to N_dom_pop
    arma::vec rand_eff_pop_dom(N_dom_pop, arma::fill::zeros);
    for (int d = 0; d < N_dom_pop; d++) {
      if (dist_obs_dom[d]) {
        int s = pop_to_smp_map[d];
        if (s >= 0 && s < N_dom_smp) {
          rand_eff_pop_dom[d] = rand_eff_b[s];
        }
      }
    }

    // Expand rand_eff to population level
    arma::vec rand_eff_pop_vec(N_pop);
    {
      int pos = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        int nd = n_pop[d];
        for (int j = 0; j < nd; j++) {
          rand_eff_pop_vec[pos++] = rand_eff_pop_dom[d];
        }
      }
    }

    // mu_fixed_b and mu_b
    arma::vec mu_fixed_b = X_pop * betas_b;
    arma::vec mu_b = mu_fixed_b + rand_eff_pop_vec;

    // =====================================================================
    // Step 9: Monte Carlo simulation (inlined)
    // =====================================================================

    double sqrt_sigmae2_b = std::sqrt(sigma2_e_b);
    double sqrt_sigmau2_b = std::sqrt(sigma2_u_b);

    // Precompute sqrt of in-sample domain variances
    arma::vec sqrt_sigmav2(N_dom_smp_selected);
    for (int i = 0; i < N_dom_smp_selected; i++) {
      sqrt_sigmav2[i] = std::sqrt(sigmav2_selected[i]);
    }

    // ------------------------------------------------------------------
    // Pre-generate all random numbers for MC loop (R::rnorm not thread-safe)
    // ------------------------------------------------------------------
    arma::mat mc_all_epsilon(N_pop, L);
    for (int l = 0; l < L; l++) {
      for (int i = 0; i < N_pop; i++) {
        mc_all_epsilon(i, l) = R::rnorm(0.0, sqrt_sigmae2_b);
      }
    }

    int mc_N_unobs = N_dom_unobs;
    arma::mat mc_all_vu_unobs(std::max(mc_N_unobs, 1), L);
    arma::mat mc_all_vu_insmp(std::max(N_dom_smp_selected, 1), L);
    for (int l = 0; l < L; l++) {
      int ui = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        if (!dist_obs_dom[d]) {
          mc_all_vu_unobs(ui++, l) = R::rnorm(0.0, sqrt_sigmau2_b);
        }
      }
      int si = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        if (dist_obs_dom[d]) {
          mc_all_vu_insmp(si, l) = R::rnorm(0.0, sqrt_sigmav2[si]);
          si++;
        }
      }
    }

    // MC indicator accumulator
    arma::mat mc_indicator_sum(N_dom_ind, n_indicators, arma::fill::zeros);

    // ------------------------------------------------------------------
    // MC loop — OpenMP parallelized over L iterations
    // ------------------------------------------------------------------
    #ifdef _OPENMP
    // num_threads scopes the count to this region: emdi2 never calls
    // omp_set_num_threads(), so it cannot alter OpenMP for the rest of the
    // session or for other packages.
    #pragma omp parallel if(L > 10) num_threads(threads)
    {
    #endif
      arma::mat local_mc_sum(N_dom_ind, n_indicators, arma::fill::zeros);
      arma::vec mc_vu_local(N_pop);

      #ifdef _OPENMP
      #pragma omp for schedule(static)
      #endif
      for (int l = 0; l < L; l++) {

        // Build vu from pre-generated values
        {
          int unobs_idx = 0, smp_sel_idx = 0, pos = 0;
          for (int d = 0; d < N_dom_pop; d++) {
            double vu_d;
            if (dist_obs_dom[d]) {
              vu_d = mc_all_vu_insmp(smp_sel_idx++, l);
            } else {
              vu_d = mc_all_vu_unobs(unobs_idx++, l);
            }
            int nd = n_pop[d];
            for (int j = 0; j < nd; j++) {
              mc_vu_local[pos++] = vu_d;
            }
          }
        }

        arma::vec y_star = mu_b + mc_all_epsilon.col(l) + mc_vu_local;
        arma::vec y_bt = back_transform_arma(y_star, transformation, lambda_b, shift_b);

        for (int i = 0; i < N_pop; i++) {
          if (!std::isfinite(y_bt[i])) y_bt[i] = 0.0;
        }

        // Compute indicators per domain
        if (use_agg) {
          for (int d = 0; d < N_dom_ind; d++) {
            const arma::uvec& idx = agg_idx_cache[d];
            arma::vec y_d = y_bt.elem(idx);
            arma::vec w_d = pop_weights.elem(idx);
            local_mc_sum.row(d) += compute_domain_indicators_selective_cpp(y_d, w_d, threshold, indicator_mask).t();
          }
        } else {
          int offset = 0;
          for (int d = 0; d < N_dom_pop; d++) {
            int nd = n_pop[d];
            arma::vec y_d = y_bt.subvec(offset, offset + nd - 1);
            arma::vec w_d = pop_weights.subvec(offset, offset + nd - 1);
            local_mc_sum.row(d) += compute_domain_indicators_selective_cpp(y_d, w_d, threshold, indicator_mask).t();
            offset += nd;
          }
        }
      } // end omp for

      #ifdef _OPENMP
      #pragma omp critical
      #endif
      {
        mc_indicator_sum += local_mc_sum;
      }

    #ifdef _OPENMP
    } // end omp parallel
    #endif

    // =====================================================================
    // Step 10: MSE accumulation
    // =====================================================================

    // boot_estimates = mean of MC indicators over L
    arma::mat boot_estimates = mc_indicator_sum / (double)L;

    // mse_accum += (boot_estimates - true_indicators)^2
    arma::mat diff = boot_estimates - true_indicators;
    mse_accum += diff % diff; // element-wise square

    // Progress: single in-place line carrying elapsed time and projected
    // finish, throttled by elapsed time rather than iteration count (a
    // bootstrap iteration can take milliseconds or minutes depending on N_pop
    // and L). Format mirrors R/progress.R -- see src/progress.h.
    {
      double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      bool final_iter = (b == B - 1);
      if (final_iter || elapsed - last_report >= 0.5) {
        last_report = elapsed;
        Rprintf("\r%s", emdi::progress_line(b + 1, B, elapsed, start_epoch,
                                            "bootstrap iteration").c_str());
        if (final_iter) Rprintf("\n");
        R_FlushConsole();
      }
    }

  } // end bootstrap loop

  // Average MSE over B
  arma::mat mse = mse_accum / (double)B;

  return mse;
}
