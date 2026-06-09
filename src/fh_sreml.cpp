#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Spatial Fay-Herriot REML via Fisher scoring over (sigma2_u, rho).
// Faithful transcription of R SREML (R/estim_sigmau2.R:405-502). Dense V.
// V = sigma2_u * [(I - rho*W')(I - rho*W)]^{-1} + diag(vardir). Dense.
// [[Rcpp::export]]
Rcpp::List fh_sreml_cpp(const arma::vec& direct, const arma::mat& X,
                        const arma::vec& vardir, const arma::mat& W,
                        int maxit, double tol) {
  const arma::uword m = direct.n_elem;
  arma::mat I  = arma::eye(m, m);
  arma::mat Wt = W.t();
  arma::mat Xt = X.t();
  arma::mat Dpsi = arma::diagmat(vardir);

  double est_sigma2 = arma::median(vardir);
  double est_rho    = 0.5;
  double conv = tol + 1.0;
  int iter = 0;

  while (conv > tol && iter < maxit) {
    iter++;
    double cur_sigma2 = est_sigma2;   // est.sigma2[iter]
    double cur_rho    = est_rho;      // est.rho[iter]

    // Derivative of V w.r.t. sigma2_u: der.sigma = solve((I - rho*W') %*% (I - rho*W))
    arma::mat der_sigma = arma::inv((I - cur_rho * Wt) * (I - cur_rho * W));

    // Derivative of V w.r.t. rho: der.vrho = -sigma2 * der.sigma %*% der.rho %*% der.sigma
    arma::mat der_rho  = 2.0 * cur_rho * Wt * W - W - Wt;
    arma::mat der_vrho = -cur_sigma2 * (der_sigma * der_rho * der_sigma);

    // Covariance matrix V and its inverse
    arma::mat V  = cur_sigma2 * der_sigma + Dpsi;
    arma::mat Vi = arma::inv(V);

    // Projection matrix P = V^-1 - V^-1 X (X' V^-1 X)^-1 X' V^-1
    arma::mat XVi = Xt * Vi;
    arma::mat Q  = arma::inv(XVi * X);
    arma::mat P  = Vi - (Vi * X * Q * XVi);

    // Scores vector
    arma::mat P_der_sigma = P * der_sigma;
    arma::mat P_der_rho   = P * der_vrho;
    arma::vec P_direct    = P * direct;

    arma::vec score(2);
    score(0) = -0.5 * arma::trace(P_der_sigma)
               + 0.5 * arma::as_scalar(direct.t() * P_der_sigma * P_direct);
    score(1) = -0.5 * arma::trace(P_der_rho)
               + 0.5 * arma::as_scalar(direct.t() * P_der_rho * P_direct);

    // Fisher information matrix
    arma::mat fisher(2, 2);
    fisher(0, 0) = 0.5 * arma::trace(P_der_sigma * P_der_sigma);
    fisher(0, 1) = 0.5 * arma::trace(P_der_sigma * P_der_rho);
    fisher(1, 0) = 0.5 * arma::trace(P_der_rho   * P_der_sigma);
    fisher(1, 1) = 0.5 * arma::trace(P_der_rho   * P_der_rho);

    // Updating equations: final = est + fisher^{-1} * score
    arma::vec est_param(2);
    est_param(0) = cur_sigma2;
    est_param(1) = cur_rho;
    arma::vec final_param = est_param + arma::solve(fisher, score);

    // Restrict rho to (-0.999, 0.999)
    if (final_param(1) <= -1.0) final_param(1) = -0.999;
    if (final_param(1) >=  1.0) final_param(1) =  0.999;

    est_sigma2 = final_param(0);
    est_rho    = final_param(1);
    conv = arma::max(arma::abs(final_param - est_param) / est_param);
  }

  // Map boundary values back to exact +/-1
  double rho = est_rho;
  if (rho == -0.999) rho = -1.0;
  else if (rho == 0.999) rho = 1.0;

  double sigma2u = std::max(est_sigma2, 0.0);
  bool convergence = !(iter >= maxit && conv >= tol);

  return Rcpp::List::create(Rcpp::Named("sigmau2")     = sigma2u,
                            Rcpp::Named("rho")         = rho,
                            Rcpp::Named("convergence") = convergence);
}
