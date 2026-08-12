#include <RcppArmadillo.h>
#include "brent.h"
// [[Rcpp::depends(RcppArmadillo)]]

// Tolerance the EBP kernels have always used for the Brent searches
// (sqrt(DBL_EPSILON)). Kept explicit and unchanged while the three
// duplicated Brent copies were consolidated onto emdi::brent_fmin, so
// the only thing that moves is the iteration path, not the precision.
static const double kBrentTol = 1.490116e-08;

// [[Rcpp::export]]
Rcpp::List data_transform_cpp(const arma::vec& y_raw,
                               const std::string& transformation,
                               double lambda) {
  arma::vec y = y_raw;
  double shift = 0.0;
  bool shift_is_null = false;

  if (transformation == "no") {
    shift_is_null = true;
  } else if (transformation == "log") {
    double mn = y.min();
    if (mn <= 0) { shift = std::abs(mn) + 1.0; y = y + shift; }
    y = arma::log(y);
  } else if (transformation == "box.cox") {
    double mn = y.min();
    if (mn <= 0) { shift = shift + std::abs(mn) + 1.0; }
    if (std::abs(lambda) <= 1e-12) { y = arma::log(y + shift); }
    else { y = (arma::pow(y + shift, lambda) - 1.0) / lambda; }
  } else if (transformation == "dual") {
    double mn = y.min();
    if (mn <= 0) { shift = shift + std::abs(mn) + 1.0; }
    if (std::abs(lambda) <= 1e-12) { y = arma::log(y + shift); }
    else {
      arma::vec yps = y + shift;
      y = (arma::pow(yps, lambda) - arma::pow(yps, -lambda)) / (2.0 * lambda);
    }
  } else if (transformation == "log.shift") {
    double mn = arma::min(y + lambda);
    if (mn <= 0) { lambda = lambda + std::abs(mn) + 1.0; }
    y = arma::log(y + lambda);
    shift_is_null = true;
  } else {
    Rcpp::stop("Unknown transformation: " + transformation);
  }

  if (shift_is_null) {
    return Rcpp::List::create(Rcpp::Named("y") = y, Rcpp::Named("shift") = R_NilValue);
  } else {
    return Rcpp::List::create(Rcpp::Named("y") = y, Rcpp::Named("shift") = shift);
  }
}

// ---------------------------------------------------------------------------
// Per-domain sufficient statistics for LME fit
// ---------------------------------------------------------------------------
struct DomainSuffStats {
  arma::mat S_xx;   // p x p: X_d' X_d
  arma::vec S_xy;   // p x 1: X_d' y_d
  double S_yy;      // y_d' y_d
  arma::vec xbar;   // p x 1: colSums(X_d)
  double ybar;      // sum(y_d)
  int nd;
};

// ---------------------------------------------------------------------------
// lme_fit_cpp: Fit random-intercept REML model, return all parameters
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List lme_fit_cpp(const arma::vec& y_transformed,
                        const arma::mat& X,
                        const arma::ivec& n_d) {

  int n = y_transformed.n_elem;
  int p = X.n_cols;
  int D = n_d.n_elem;

  // Defensive: n_d must contain strictly positive counts (see reml_loglik_cpp
  // for the rationale — empty domains trigger Armadillo's X.rows(off, off-1)).
  if (arma::any(n_d <= 0)) {
    Rcpp::stop("lme_fit_cpp: n_d contains non-positive counts. "
               "Drop unused factor levels from smp_domains before calling.");
  }

  // 1. Precompute per-domain sufficient statistics
  std::vector<DomainSuffStats> stats(D);
  int offset = 0;
  for (int d = 0; d < D; ++d) {
    int nd = n_d(d);
    arma::mat X_d = X.rows(offset, offset + nd - 1);
    arma::vec y_d = y_transformed.subvec(offset, offset + nd - 1);

    stats[d].nd = nd;
    stats[d].S_xx = X_d.t() * X_d;
    stats[d].S_xy = X_d.t() * y_d;
    stats[d].S_yy = arma::dot(y_d, y_d);
    stats[d].xbar = arma::sum(X_d, 0).t();  // column sums
    stats[d].ybar = arma::sum(y_d);

    offset += nd;
  }

  // 2. Define negative REML profile log-likelihood as function of log(theta)
  auto neg_reml = [&](double log_theta) -> double {
    double theta = std::exp(log_theta);

    arma::mat A(p, p, arma::fill::zeros);
    arma::vec b_vec(p, arma::fill::zeros);
    double log_det_V_part = 0.0;

    for (int d = 0; d < D; ++d) {
      double c_d = theta / (1.0 + stats[d].nd * theta);
      A += stats[d].S_xx - c_d * (stats[d].xbar * stats[d].xbar.t());
      b_vec += stats[d].S_xy - c_d * stats[d].xbar * stats[d].ybar;
      log_det_V_part += std::log(1.0 + stats[d].nd * theta);
    }

    arma::vec beta;
    bool solved = arma::solve(beta, A, b_vec, arma::solve_opts::likely_sympd);
    if (!solved) return 1e30;

    double quad = 0.0;
    for (int d = 0; d < D; ++d) {
      double c_d = theta / (1.0 + stats[d].nd * theta);
      double rss_d = stats[d].S_yy
        - 2.0 * arma::dot(beta, stats[d].S_xy)
        + arma::as_scalar(beta.t() * stats[d].S_xx * beta);
      double rbar_d = stats[d].ybar - arma::dot(stats[d].xbar, beta);
      quad += rss_d - c_d * rbar_d * rbar_d;
    }

    double sigma2_e = quad / (n - p);
    if (sigma2_e <= 0) return 1e30;

    double log_det_V = n * std::log(sigma2_e) + log_det_V_part;

    double log_det_A_val, log_det_A_sign;
    arma::log_det(log_det_A_val, log_det_A_sign, A);
    if (log_det_A_sign <= 0) return 1e30;
    double log_det_XVX = log_det_A_val - p * std::log(sigma2_e);

    double neg2_reml = (n - p) * std::log(2.0 * M_PI)
      + log_det_V + log_det_XVX + (n - p);

    return 0.5 * neg2_reml;
  };

  // 3. Optimize over log(theta) using Brent's method in [-15, 15]
  double best_log_theta = emdi::brent_fmin(-15.0, 15.0, neg_reml, kBrentTol);
  double theta_hat = std::exp(best_log_theta);

  // 4. Recover all parameters at optimal theta
  arma::mat A(p, p, arma::fill::zeros);
  arma::vec b_vec(p, arma::fill::zeros);

  for (int d = 0; d < D; ++d) {
    double c_d = theta_hat / (1.0 + stats[d].nd * theta_hat);
    A += stats[d].S_xx - c_d * (stats[d].xbar * stats[d].xbar.t());
    b_vec += stats[d].S_xy - c_d * stats[d].xbar * stats[d].ybar;
  }

  arma::vec betas = arma::solve(A, b_vec, arma::solve_opts::likely_sympd);

  double quad = 0.0;
  for (int d = 0; d < D; ++d) {
    double c_d = theta_hat / (1.0 + stats[d].nd * theta_hat);
    double rss_d = stats[d].S_yy
      - 2.0 * arma::dot(betas, stats[d].S_xy)
      + arma::as_scalar(betas.t() * stats[d].S_xx * betas);
    double rbar_d = stats[d].ybar - arma::dot(stats[d].xbar, betas);
    quad += rss_d - c_d * rbar_d * rbar_d;
  }

  double sigma2_e = quad / (n - p);
  double sigma2_u = theta_hat * sigma2_e;

  // 5. Compute BLUPs and gamma (shrinkage factors)
  arma::vec rand_eff(D);
  arma::vec gamma(D);

  for (int d = 0; d < D; ++d) {
    double gamma_d = sigma2_u / (sigma2_u + sigma2_e / stats[d].nd);
    gamma(d) = gamma_d;
    // mean(y_d) - mean(X_d)' * beta
    double mean_y_d = stats[d].ybar / stats[d].nd;
    arma::vec mean_X_d = stats[d].xbar / stats[d].nd;
    rand_eff(d) = gamma_d * (mean_y_d - arma::dot(mean_X_d, betas));
  }

  return Rcpp::List::create(
    Rcpp::Named("betas") = betas,
    Rcpp::Named("sigma2_e") = sigma2_e,
    Rcpp::Named("sigma2_u") = sigma2_u,
    Rcpp::Named("rand_eff") = rand_eff,
    Rcpp::Named("gamma") = gamma
  );
}

// ---------------------------------------------------------------------------
// model_par_weighted_cpp: Compute pseudo-EB weighted model parameters
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List model_par_weighted_cpp(const arma::vec& y_transformed,
                                   const arma::mat& X,
                                   const arma::vec& weights,
                                   const arma::ivec& n_d,
                                   double sigma2_e,
                                   double sigma2_u) {
  int D = n_d.n_elem;
  int p = X.n_cols;

  // Defensive: see reml_loglik_cpp for rationale (empty domains crash X.rows).
  if (arma::any(n_d <= 0)) {
    Rcpp::stop("model_par_weighted_cpp: n_d contains non-positive counts. "
               "Drop unused factor levels from smp_domains before calling.");
  }

  arma::vec weight_sum(D);
  arma::vec mean_dep(D);
  arma::mat mean_indep(D, p);
  arma::vec delta2(D);
  arma::vec gammaw(D);

  arma::vec num_vec(p, arma::fill::zeros);
  arma::mat den_mat(p, p, arma::fill::zeros);

  int offset = 0;
  for (int d = 0; d < D; ++d) {
    int nd = n_d(d);
    arma::mat X_d = X.rows(offset, offset + nd - 1);
    arma::vec y_d = y_transformed.subvec(offset, offset + nd - 1);
    arma::vec w_d = weights.subvec(offset, offset + nd - 1);

    // Weight sums
    weight_sum(d) = arma::sum(w_d);
    mean_dep(d) = arma::dot(w_d, y_d) / weight_sum(d);

    // Weighted means of predictors
    for (int k = 0; k < p; ++k) {
      mean_indep(d, k) = arma::dot(w_d, X_d.col(k)) / weight_sum(d);
    }

    delta2(d) = arma::dot(w_d, w_d) / (weight_sum(d) * weight_sum(d));
    gammaw(d) = sigma2_u / (sigma2_u + sigma2_e * delta2(d));

    // dep_var_ast = y_d - gamma_weight[d] * mean_dep[d]
    arma::vec dep_var_ast = y_d - gammaw(d) * mean_dep(d);

    // indep_weight = t(X_d) %*% diag(w_d)  =>  each col of X_d scaled by w_d
    arma::mat indep_weight = X_d.each_col() % w_d;  // nd x p, rows scaled by w_d
    // indep_weight is X_d with rows scaled; we need t(X_d) %*% diag(w_d) = (X_d .* w_d).t()
    arma::mat indep_weight_t = indep_weight.t();  // p x nd

    // indep_var_ast = X_d - outer(ones, gamma_weight[d] * mean_indep[d,])
    arma::vec gw_mean_indep = gammaw(d) * mean_indep.row(d).t();  // p x 1
    arma::mat indep_var_ast = X_d.each_row() - gw_mean_indep.t();  // nd x p

    // num += indep_weight_t %*% dep_var_ast  (p x 1)
    num_vec += indep_weight_t * dep_var_ast;

    // den += indep_weight_t %*% indep_var_ast  (p x p)
    den_mat += indep_weight_t * indep_var_ast;

    offset += nd;
  }

  // betas = solve(den, num)
  arma::vec betas = arma::solve(den_mat, num_vec);

  // rand_eff[d] = gamma_weight[d] * (mean_dep[d] - mean_indep[d,] %*% betas)
  arma::vec rand_eff(D);
  for (int d = 0; d < D; ++d) {
    rand_eff(d) = gammaw(d) * (mean_dep(d) - arma::dot(mean_indep.row(d).t(), betas));
  }

  return Rcpp::List::create(
    Rcpp::Named("betas") = betas,
    Rcpp::Named("rand_eff") = rand_eff,
    Rcpp::Named("gammaw") = gammaw,
    Rcpp::Named("delta2") = delta2
  );
}
