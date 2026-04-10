#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Forward declarations for functions defined in other translation units.
// back_transform_cpp returns Rcpp::NumericVector (see transformations.cpp).
Rcpp::NumericVector back_transform_cpp(const arma::vec& y,
                                       const std::string& transformation,
                                       double lambda,
                                       double shift);

// compute_domain_indicators_cpp returns arma::vec of length 10 (see indicators.cpp).
arma::vec compute_domain_indicators_cpp(const arma::vec& y,
                                        const arma::vec& weights,
                                        double threshold);

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
    int               N_dom_agg = 0
) {
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
    for (int d = 0; d < N_dom_ind; d++) {
      agg_idx_cache[d] = arma::find(agg_ids == (d + 1));
    }
  }

  // Accumulator for indicator sums across iterations [N_dom_ind x n_indicators]
  arma::mat indicator_sum(N_dom_ind, n_indicators, arma::fill::zeros);

  // Storage for all MC populations [N_pop x L]
  arma::mat y_mcmc(N_pop, L);

  // Working vectors
  arma::vec epsilon(N_pop);
  arma::vec vu(N_pop);

  for (int l = 0; l < L; l++) {

    // ------------------------------------------------------------------
    // Step 1: Generate epsilon for all N_pop units (preserves RNG order)
    // ------------------------------------------------------------------
    for (int i = 0; i < N_pop; i++) {
      epsilon(i) = R::rnorm(0.0, sqrt_sigmae2);
    }

    // ------------------------------------------------------------------
    // Step 2a: Generate out-of-sample domain effects (N_dom_unobs draws)
    // ------------------------------------------------------------------
    arma::vec vu_unobs(N_dom_unobs);
    {
      int unobs_idx = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        if (!dist_obs_dom(d)) {
          vu_unobs(unobs_idx++) = R::rnorm(0.0, sqrt_sigmau2);
        }
      }
    }

    // ------------------------------------------------------------------
    // Step 2b: Generate in-sample domain effects (N_dom_smp draws)
    // ------------------------------------------------------------------
    arma::vec vu_insmp(N_dom_smp);
    {
      int smp_idx = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        if (dist_obs_dom(d)) {
          vu_insmp(smp_idx) = R::rnorm(0.0, sqrt_sigmav2(smp_idx));
          smp_idx++;
        }
      }
    }

    // ------------------------------------------------------------------
    // Step 3: Assign vu to population vector and compute y = mu + eps + vu
    // ------------------------------------------------------------------
    {
      int offset    = 0;
      int unobs_idx = 0;
      int smp_idx   = 0;

      for (int d = 0; d < N_dom_pop; d++) {
        double vu_d;
        if (dist_obs_dom(d)) {
          vu_d = vu_insmp(smp_idx++);
        } else {
          vu_d = vu_unobs(unobs_idx++);
        }

        int nd = n_pop(d);
        for (int i = 0; i < nd; i++) {
          vu(offset + i) = vu_d;
        }
        offset += nd;
      }
    }

    // y on transformed scale
    arma::vec y_star = mu + epsilon + vu;

    // ------------------------------------------------------------------
    // Step 4: Back-transform
    // ------------------------------------------------------------------
    Rcpp::NumericVector y_bt_rcpp = back_transform_cpp(y_star, transformation, lambda, shift);
    arma::vec y_bt = Rcpp::as<arma::vec>(y_bt_rcpp);

    // ------------------------------------------------------------------
    // Step 5: Replace non-finite values with 0
    // ------------------------------------------------------------------
    for (int i = 0; i < N_pop; i++) {
      if (!std::isfinite(y_bt(i))) {
        y_bt(i) = 0.0;
      }
    }

    // Store this iteration's back-transformed population
    y_mcmc.col(l) = y_bt;

    // ------------------------------------------------------------------
    // Step 6: Compute indicators per domain and accumulate
    // ------------------------------------------------------------------
    if (use_agg) {
      // Aggregated domains: observations may be non-contiguous
      for (int d = 0; d < N_dom_ind; d++) {
        const arma::uvec& idx = agg_idx_cache[d];
        arma::vec y_d = y_bt.elem(idx);
        arma::vec w_d = pop_weights.elem(idx);
        arma::vec ind = compute_domain_indicators_cpp(y_d, w_d, threshold);
        indicator_sum.row(d) += ind.t();
      }
    } else {
      // Standard: domains are contiguous blocks
      int offset = 0;
      for (int d = 0; d < N_dom_pop; d++) {
        int nd = n_pop(d);
        arma::vec y_d = y_bt.subvec(offset, offset + nd - 1);
        arma::vec w_d = pop_weights.subvec(offset, offset + nd - 1);
        arma::vec ind = compute_domain_indicators_cpp(y_d, w_d, threshold);
        indicator_sum.row(d) += ind.t();
        offset += nd;
      }
    }

  } // end L iterations

  // Average over iterations
  arma::mat point_estimates = indicator_sum / (double)L;

  return Rcpp::List::create(
    Rcpp::Named("point_estimates") = point_estimates,
    Rcpp::Named("y_mcmc")          = y_mcmc
  );
}
