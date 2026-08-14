#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#ifdef _OPENMP
  #include <omp.h>
#endif

// Forward declarations for functions defined in other translation units.
// back_transform_arma returns arma::vec directly (no copy).
arma::vec back_transform_arma(const arma::vec& y,
                               const std::string& transformation,
                               double lambda, double shift);

// back_transform_cpp returns Rcpp::NumericVector (kept for R-facing API).
Rcpp::NumericVector back_transform_cpp(const arma::vec& y,
                                       const std::string& transformation,
                                       double lambda, double shift);

// compute_domain_indicators_cpp returns arma::vec of length 10 (see indicators.cpp).
arma::vec compute_domain_indicators_cpp(const arma::vec& y,
                                        const arma::vec& weights,
                                        double threshold);
arma::vec compute_domain_indicators_selective_cpp(const arma::vec& y,
                                                    const arma::vec& weights,
                                                    double threshold,
                                                    int indicator_mask);

// ---------------------------------------------------------------------------
// monte_carlo_cpp
//
// Runs L Monte Carlo iterations, each time:
//   1. Drawing individual errors epsilon ~ N(0, sigmae2) for all N_pop obs.
//   2. Drawing random effects vu per domain:
//        - Out-of-sample domains: vu ~ N(0, sigmau2)
//        - In-sample domains:     vu ~ N(0, sigmav2[smp_idx])
//   3. Predicting y = mu + epsilon + vu (on the transformed scale).
//   4. Back-transforming y.
//   5. Replacing non-finite values with 0.
//   6. Computing all 10 welfare indicators per domain (or aggregated domain).
//
// RNG ORDER (must match R errors_gen() exactly for reproducibility):
//   a. N_pop draws for epsilon (loop over all population units).
//   b. N_dom_unobs draws for out-of-sample domain effects (order in dist_obs_dom).
//   c. N_dom_smp draws for in-sample domain effects (order in dist_obs_dom).
//
// Parameters
// ----------
// mu            : constant part (X*beta + fitted random effect) [N_pop]
// sigmae2       : individual error variance
// sigmau2       : variance for out-of-sample domain random effects
// sigmav2       : domain-specific variance for in-sample RE [N_dom_smp]
// domain_ids    : 1-based domain index for each population unit [N_pop]
// obs_dom       : 1 if unit is in an in-sample domain, else 0 [N_pop]
// dist_obs_dom  : 1 if domain is in-sample, else 0 [N_dom_pop]
// n_pop         : number of population units per domain [N_dom_pop]
// N_dom_pop     : total number of domains in population
// N_dom_smp     : number of in-sample domains
// N_dom_unobs   : number of out-of-sample domains
// L             : number of Monte Carlo iterations
// threshold     : poverty line (for HCR, PGap computation)
// transformation: name of the back-transformation ("no", "log", ...)
// lambda        : Box-Cox / dual parameter
// shift         : shift parameter
// pop_weights   : survey/calibration weights [N_pop]
// n_indicators  : number of indicators (should be 10)
// agg_domain_ids: (optional) 1-based aggregated domain index per obs [N_pop].
//                 When provided, indicators are computed on aggregated domains.
// N_dom_agg     : (optional) number of aggregated domains. Required when
//                 agg_domain_ids is provided.
//
// Returns
// -------
// A named List with:
//   "point_estimates" : arma::mat [N_dom x n_indicators] — mean over L iters
//                       (N_dom = N_dom_agg when aggregate, else N_dom_pop)
//   "y_mcmc"          : arma::mat [N_pop x L] — back-transformed y per iter
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List monte_carlo_cpp(
    const arma::vec&  mu,
    double            sigmae2,
    double            sigmau2,
    const arma::vec&  sigmav2,
    const arma::ivec& domain_ids,
    const arma::ivec& obs_dom,
    const arma::ivec& dist_obs_dom,
    const arma::ivec& n_pop,
    int               N_dom_pop,
    int               N_dom_smp,
    int               N_dom_unobs,
    int               L,
    double            threshold,
    const std::string& transformation,
    double            lambda,
    double            shift,
    const arma::vec&  pop_weights,
    int               n_indicators,
    Rcpp::Nullable<Rcpp::IntegerVector> agg_domain_ids = R_NilValue,
    int               N_dom_agg = 0,
    int               indicator_mask = 0x3FF,
    int               threads = 1
) {
  if (threads < 1) threads = 1;

  int N_pop = (int)mu.n_elem;

  double sqrt_sigmae2 = std::sqrt(sigmae2);
  double sqrt_sigmau2 = std::sqrt(sigmau2);

  // Precompute sqrt of in-sample domain variances
  arma::vec sqrt_sigmav2(N_dom_smp);
  for (int s = 0; s < N_dom_smp; s++) {
    sqrt_sigmav2(s) = std::sqrt(sigmav2(s));
  }

  // Determine whether we aggregate indicators to a coarser domain level
  bool use_agg = agg_domain_ids.isNotNull() && N_dom_agg > 0;
  arma::ivec agg_ids;
  int N_dom_ind; // number of domains for indicator output
  if (use_agg) {
    agg_ids = Rcpp::as<arma::ivec>(agg_domain_ids.get());
    N_dom_ind = N_dom_agg;
  } else {
    N_dom_ind = N_dom_pop;
  }

  // Pre-build index vectors for aggregated domains (observations per agg domain)
  // so we don't call arma::find() every iteration
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

  // Accumulator for indicator sums across iterations [N_dom_ind x n_indicators]
  arma::mat indicator_sum(N_dom_ind, n_indicators, arma::fill::zeros);

  // Storage for all MC populations [N_pop x L]
  arma::mat y_mcmc(N_pop, L);

  // ------------------------------------------------------------------
  // Pre-generate ALL random numbers sequentially (R::rnorm is not thread-safe).
  // IMPORTANT: Generate one iteration at a time to preserve the original RNG
  // stream order (eps_l, vu_unobs_l, vu_insmp_l for each l).
  // ------------------------------------------------------------------

  arma::mat all_epsilon(N_pop, L);
  arma::mat all_vu_unobs(std::max(N_dom_unobs, 1), L);
  arma::mat all_vu_insmp(std::max(N_dom_smp, 1), L);

  for (int l = 0; l < L; l++) {
    // Step 1: epsilon (N_pop draws)
    for (int i = 0; i < N_pop; i++) {
      all_epsilon(i, l) = R::rnorm(0.0, sqrt_sigmae2);
    }
    // Step 2: out-of-sample vu (N_dom_unobs draws)
    int unobs_idx = 0;
    for (int d = 0; d < N_dom_pop; d++) {
      if (!dist_obs_dom(d)) {
        all_vu_unobs(unobs_idx++, l) = R::rnorm(0.0, sqrt_sigmau2);
      }
    }
    // Step 3: in-sample vu (N_dom_smp draws)
    int smp_idx = 0;
    for (int d = 0; d < N_dom_pop; d++) {
      if (dist_obs_dom(d)) {
        all_vu_insmp(smp_idx, l) = R::rnorm(0.0, sqrt_sigmav2(smp_idx));
        smp_idx++;
      }
    }
  }

  // Precompute domain offsets for contiguous blocks
  std::vector<int> dom_offset(N_dom_pop);
  {
    int off = 0;
    for (int d = 0; d < N_dom_pop; d++) {
      dom_offset[d] = off;
      off += n_pop(d);
    }
  }

  // ------------------------------------------------------------------
  // MC loop — OpenMP parallelized over L iterations.
  // Each iteration is independent: reads pre-generated RNG, writes to
  // its own column of y_mcmc, accumulates into thread-local indicator sums.
  // ------------------------------------------------------------------
  #ifdef _OPENMP
  // num_threads scopes the count to this region: emdi never calls
  // omp_set_num_threads(), so it cannot alter OpenMP for the rest of the
  // session or for other packages.
  #pragma omp parallel if(L > 10) num_threads(threads)
  {
  #endif
    // Thread-local indicator accumulator
    arma::mat local_sum(N_dom_ind, n_indicators, arma::fill::zeros);
    arma::vec vu(N_pop);  // thread-local working buffer

    #ifdef _OPENMP
    #pragma omp for schedule(static)
    #endif
    for (int l = 0; l < L; l++) {

      // Build vu vector from pre-generated domain effects
      {
        int unobs_idx = 0;
        int smp_idx = 0;
        for (int d = 0; d < N_dom_pop; d++) {
          double vu_d;
          if (dist_obs_dom(d)) {
            vu_d = all_vu_insmp(smp_idx++, l);
          } else {
            vu_d = all_vu_unobs(unobs_idx++, l);
          }
          int nd = n_pop(d);
          int off = dom_offset[d];
          for (int i = 0; i < nd; i++) {
            vu(off + i) = vu_d;
          }
        }
      }

      // y on transformed scale: y_star = mu + epsilon + vu
      arma::vec y_star = mu + all_epsilon.col(l) + vu;

      // Back-transform (using arma::vec version — no Rcpp copy)
      arma::vec y_bt = back_transform_arma(y_star, transformation, lambda, shift);

      // Replace non-finite values with 0
      for (int i = 0; i < N_pop; i++) {
        if (!std::isfinite(y_bt(i))) y_bt(i) = 0.0;
      }

      // Store MC population
      y_mcmc.col(l) = y_bt;

      // Compute indicators per domain and accumulate to thread-local sum
      if (use_agg) {
        for (int d = 0; d < N_dom_ind; d++) {
          const arma::uvec& idx = agg_idx_cache[d];
          arma::vec y_d = y_bt.elem(idx);
          arma::vec w_d = pop_weights.elem(idx);
          arma::vec ind = compute_domain_indicators_selective_cpp(y_d, w_d, threshold, indicator_mask);
          local_sum.row(d) += ind.t();
        }
      } else {
        for (int d = 0; d < N_dom_pop; d++) {
          int nd = n_pop(d);
          int off = dom_offset[d];
          arma::vec y_d = y_bt.subvec(off, off + nd - 1);
          arma::vec w_d = pop_weights.subvec(off, off + nd - 1);
          arma::vec ind = compute_domain_indicators_selective_cpp(y_d, w_d, threshold, indicator_mask);
          local_sum.row(d) += ind.t();
        }
      }
    } // end parallel for

    // Reduce thread-local sums into global accumulator
    #ifdef _OPENMP
    #pragma omp critical
    #endif
    {
      indicator_sum += local_sum;
    }

  #ifdef _OPENMP
  } // end omp parallel
  #endif

  // Average over iterations
  arma::mat point_estimates = indicator_sum / (double)L;

  return Rcpp::List::create(
    Rcpp::Named("point_estimates") = point_estimates,
    Rcpp::Named("y_mcmc")          = y_mcmc
  );
}
