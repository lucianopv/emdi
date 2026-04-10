#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Forward declaration for back_transform_cpp defined in transformations.cpp.
// Returns Rcpp::NumericVector.
Rcpp::NumericVector back_transform_cpp(const arma::vec& y,
                                       const std::string& transformation,
                                       double lambda,
                                       double shift);

// ---------------------------------------------------------------------------
// gen_superpop_cpp
//
// Generates a bootstrap superpopulation vector of income (back-transformed).
// Matches the R function superpopulation() in R/mse_estimation.R lines 307-335.
//
// RNG ORDER (must match R exactly):
//   1. rnorm(sum(obs_dom), 0, sqrt(sigmae2))       -- eps for in-sample obs
//   2. rnorm(sum(!obs_dom), 0, sqrt(sigmae2+sigmau2)) -- eps for out-of-sample obs
//   3. rnorm(N_dom_pop, 0, sqrt(sigmau2))           -- vu_tmp for domain REs
//
// Parameters
// ----------
// mu_fixed       : X_pop * betas [N_pop]
// sigmae2        : individual error variance
// sigmau2        : domain random effect variance
// obs_dom        : [N_pop] 1 if unit is in an in-sample domain, else 0
// n_pop          : per-domain population count [N_dom_pop]
// N_dom_pop      : total number of domains in population
// transformation : back-transformation name
// lambda         : Box-Cox / dual parameter
// shift          : shift parameter
//
// Returns: list(pop_income_vector, vu_tmp, eps, vu_pop, Y_pop_b_notrans)
// [[Rcpp::export]]
Rcpp::List gen_superpop_cpp(
    const arma::vec& mu_fixed,
    double sigmae2,
    double sigmau2,
    const arma::ivec& obs_dom,
    const arma::ivec& n_pop,
    int N_dom_pop,
    const std::string& transformation,
    double lambda,
    double shift
) {
  int N_pop = mu_fixed.n_elem;
  double sd_e       = std::sqrt(sigmae2);
  double sd_e_unobs = std::sqrt(sigmae2 + sigmau2);
  double sd_u       = std::sqrt(sigmau2);

  arma::vec eps(N_pop);

  // Draw eps for in-sample observations first (RNG step 1)
  for (int i = 0; i < N_pop; ++i) {
    if (obs_dom[i] != 0) {
      eps[i] = R::rnorm(0.0, sd_e);
    }
  }
  // Draw eps for out-of-sample observations second (RNG step 2)
  for (int i = 0; i < N_pop; ++i) {
    if (obs_dom[i] == 0) {
      eps[i] = R::rnorm(0.0, sd_e_unobs);
    }
  }

  // Draw domain random effects vu_tmp (RNG step 3)
  arma::vec vu_tmp(N_dom_pop);
  for (int d = 0; d < N_dom_pop; ++d) {
    vu_tmp[d] = R::rnorm(0.0, sd_u);
  }

  // Expand vu_tmp to population level: rep(vu_tmp, n_pop)
  arma::vec vu_pop(N_pop);
  int pos = 0;
  for (int d = 0; d < N_dom_pop; ++d) {
    int nd = n_pop[d];
    for (int j = 0; j < nd; ++j) {
      vu_pop[pos++] = vu_tmp[d];
    }
  }

  // Compute Y on transformed scale
  arma::vec Y_pop_b_notrans = mu_fixed + eps + vu_pop;

  // Back-transform
  Rcpp::NumericVector Y_pop_b_rv = back_transform_cpp(Y_pop_b_notrans,
                                                       transformation,
                                                       lambda, shift);
  arma::vec Y_pop_b = Rcpp::as<arma::vec>(Y_pop_b_rv);

  // Replace non-finite values with 0
  for (int i = 0; i < N_pop; ++i) {
    if (!std::isfinite(Y_pop_b[i])) Y_pop_b[i] = 0.0;
  }

  return Rcpp::List::create(
    Rcpp::Named("pop_income_vector") = Y_pop_b,
    Rcpp::Named("vu_tmp")            = vu_tmp,
    Rcpp::Named("eps")               = eps,
    Rcpp::Named("vu_pop")            = vu_pop,
    Rcpp::Named("Y_pop_b_notrans")   = Y_pop_b_notrans
  );
}

// ---------------------------------------------------------------------------
// gen_bootstrap_sample_cpp
//
// Generates bootstrap sample response values.
// Matches the R function bootstrap_par() in R/mse_estimation.R lines 339-392.
//
// RNG ORDER (must match R exactly):
//   1. rnorm(N_smp, 0, sqrt(sigmae2))   -- N_smp individual errors
//   2. For each sample domain NOT found in smp_to_pop_map: rnorm(1, 0, sqrt(sigmau2))
//
// Parameters
// ----------
// X_smp          : design matrix [N_smp x p]
// betas          : coefficient vector [p]
// sigmae2        : individual error variance
// sigmau2        : domain random effect variance
// vu_tmp         : domain random effects from superpopulation [N_dom_pop]
// smp_to_pop_map : maps sample domain index (0-based) to pop domain index (1-based).
//                  NA_integer_ if the sample domain is not in the population domains.
// n_smp          : sample unit count per sample domain [N_dom_smp]
// transformation : back-transformation name
// lambda         : Box-Cox / dual parameter
// shift          : shift parameter
//
// Returns: Y_smp_b vector [N_smp] (back-transformed bootstrap income)
// [[Rcpp::export]]
arma::vec gen_bootstrap_sample_cpp(
    const arma::mat& X_smp,
    const arma::vec& betas,
    double sigmae2,
    double sigmau2,
    const arma::vec& vu_tmp,
    const Rcpp::IntegerVector& smp_to_pop_map,
    const arma::ivec& n_smp,
    const std::string& transformation,
    double lambda,
    double shift
) {
  int N_smp = X_smp.n_rows;
  int N_dom_smp = n_smp.n_elem;
  double sd_e = std::sqrt(sigmae2);
  double sd_u = std::sqrt(sigmau2);

  // RNG step 1: draw N_smp individual errors
  arma::vec eps(N_smp);
  for (int i = 0; i < N_smp; ++i) {
    eps[i] = R::rnorm(0.0, sd_e);
  }

  // Build vu_for_smp: one RE per sample domain
  // RNG step 2: if a sample domain is not found in pop domains, draw a new RE
  arma::vec vu_for_smp(N_dom_smp);
  for (int i = 0; i < N_dom_smp; ++i) {
    int pop_idx = smp_to_pop_map[i];  // 1-based or NA_INTEGER
    if (pop_idx != NA_INTEGER) {
      // 1-based index into vu_tmp
      vu_for_smp[i] = vu_tmp[pop_idx - 1];
    } else {
      vu_for_smp[i] = R::rnorm(0.0, sd_u);
    }
  }

  // Expand vu_for_smp to sample level: rep(vu_for_smp, n_smp)
  arma::vec vu_smp(N_smp);
  int pos = 0;
  for (int i = 0; i < N_dom_smp; ++i) {
    int nd = n_smp[i];
    for (int j = 0; j < nd; ++j) {
      vu_smp[pos++] = vu_for_smp[i];
    }
  }

  // Compute Y on transformed scale: X_smp * betas + eps + vu_smp
  arma::vec mu_smp = X_smp * betas;
  arma::vec Y_smp_b_notrans = mu_smp + eps + vu_smp;

  // Back-transform
  Rcpp::NumericVector Y_smp_b_rv = back_transform_cpp(Y_smp_b_notrans,
                                                       transformation,
                                                       lambda, shift);
  arma::vec Y_smp_b = Rcpp::as<arma::vec>(Y_smp_b_rv);

  // Replace non-finite values with 0
  for (int i = 0; i < N_smp; ++i) {
    if (!std::isfinite(Y_smp_b[i])) Y_smp_b[i] = 0.0;
  }

  return Y_smp_b;
}
