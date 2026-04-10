#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

static double geometric_mean(const arma::vec& x) {
  return std::exp(arma::mean(arma::log(x)));
}

// [[Rcpp::export]]
arma::vec std_transform_y_cpp(const arma::vec& y_raw,
                               const std::string& transformation,
                               double lambda) {
  int n = y_raw.n_elem;
  arma::vec y = y_raw;  // working copy

  if (transformation == "box.cox") {
    double mn = y.min();
    if (mn <= 0) { y = y - mn + 1.0; }
    double gm = geometric_mean(y);
    arma::vec result(n);
    if (std::abs(lambda) > 1e-12) {
      double scale = lambda * std::pow(gm, lambda - 1.0);
      result = (arma::pow(y, lambda) - 1.0) / scale;
    } else {
      result = gm * arma::log(y);
    }
    return result;

  } else if (transformation == "dual") {
    double mn = y.min();
    if (mn <= 0) { y = y - mn + 1.0; }
    if (std::abs(lambda) > 1e-12) {
      arma::vec yt = (arma::pow(y, lambda) - arma::pow(y, -lambda)) / (2.0 * lambda);
      double geo = geometric_mean(arma::pow(y, lambda - 1.0) + arma::pow(y, -lambda - 1.0));
      return yt * 2.0 / geo;
    } else {
      double gm = geometric_mean(y);
      return gm * arma::log(y);
    }

  } else if (transformation == "log.shift") {
    double mn = arma::min(y + lambda);
    if (mn <= 0) {
      lambda = lambda + std::abs(y.min()) + 1.0;
    }
    arma::vec ypl = y + lambda;
    double gm = geometric_mean(ypl);
    return gm * arma::log(ypl);

  } else {
    Rcpp::stop("Unknown transformation for std_transform: " + transformation);
    return y;
  }
}

// ---------------------------------------------------------------------------
// Brent's method for 1D minimization on [a, b]
// Returns the x that minimizes f(x).
// ---------------------------------------------------------------------------
template <typename F>
static double brent_minimize(F f, double a, double b,
                             double tol = 1.490116e-08,
                             int maxiter = 1000) {
  const double golden = 0.3819660112501051;  // (3 - sqrt(5)) / 2
  double x = a + golden * (b - a);
  double w = x, v = x;
  double fx = f(x), fw = fx, fv = fx;
  double d = 0.0, e = 0.0;

  for (int iter = 0; iter < maxiter; ++iter) {
    double midpoint = 0.5 * (a + b);
    double tol1 = tol * std::abs(x) + 1e-10;
    double tol2 = 2.0 * tol1;

    if (std::abs(x - midpoint) <= (tol2 - 0.5 * (b - a))) {
      return x;
    }

    double p = 0.0, q = 0.0, r = 0.0;
    if (std::abs(e) > tol1) {
      // Parabolic interpolation
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p; else q = -q;
      r = e;
      e = d;
    }

    if (std::abs(p) < std::abs(0.5 * q * r) &&
        p > q * (a - x) && p < q * (b - x)) {
      // Parabolic step
      d = p / q;
      double u = x + d;
      if ((u - a) < tol2 || (b - u) < tol2) {
        d = (x < midpoint) ? tol1 : -tol1;
      }
    } else {
      // Golden section step
      e = (x < midpoint) ? b - x : a - x;
      d = golden * e;
    }

    double u;
    if (std::abs(d) >= tol1) {
      u = x + d;
    } else {
      u = x + ((d > 0) ? tol1 : -tol1);
    }
    double fu = f(u);

    if (fu <= fx) {
      if (u < x) b = x; else a = x;
      v = w; fv = fw;
      w = x; fw = fx;
      x = u; fx = fu;
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w; fv = fw;
        w = u; fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u; fv = fu;
      }
    }
  }
  return x;
}

// ---------------------------------------------------------------------------
// Profile REML negative log-likelihood at a given lambda
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
double reml_loglik_cpp(double lambda,
                       const arma::vec& y_raw,
                       const arma::mat& X,
                       const arma::ivec& domain_ids,
                       const arma::ivec& n_d,
                       const std::string& transformation) {

  // 1. Transform y
  arma::vec y = std_transform_y_cpp(y_raw, transformation, lambda);

  int n = y.n_elem;
  int p = X.n_cols;
  int D = n_d.n_elem;

  // 2. Precompute per-domain sufficient statistics
  struct DomainStats {
    arma::mat S_xx;   // p x p
    arma::vec S_xy;   // p x 1
    double S_yy;
    arma::vec xbar;   // p x 1  (column sums of X_d)
    double ybar;      // sum of y_d
    int nd;
  };

  std::vector<DomainStats> stats(D);

  int offset = 0;
  for (int d = 0; d < D; ++d) {
    int nd = n_d(d);
    arma::mat X_d = X.rows(offset, offset + nd - 1);
    arma::vec y_d = y.subvec(offset, offset + nd - 1);

    stats[d].nd = nd;
    stats[d].S_xx = X_d.t() * X_d;
    stats[d].S_xy = X_d.t() * y_d;
    stats[d].S_yy = arma::dot(y_d, y_d);
    stats[d].xbar = arma::sum(X_d, 0).t();  // column sums
    stats[d].ybar = arma::sum(y_d);

    offset += nd;
  }

  // 3. Define function of log(theta) that returns neg REML log-lik
  auto neg_reml = [&](double log_theta) -> double {
    double theta = std::exp(log_theta);

    // Accumulate A and b_vec
    arma::mat A(p, p, arma::fill::zeros);
    arma::vec b_vec(p, arma::fill::zeros);

    double log_det_V_part = 0.0;  // sum_d log(1 + n_d * theta)

    for (int d = 0; d < D; ++d) {
      double c_d = theta / (1.0 + stats[d].nd * theta);
      A += stats[d].S_xx - c_d * (stats[d].xbar * stats[d].xbar.t());
      b_vec += stats[d].S_xy - c_d * stats[d].xbar * stats[d].ybar;
      log_det_V_part += std::log(1.0 + stats[d].nd * theta);
    }

    // Solve for beta
    arma::vec beta;
    bool solved = arma::solve(beta, A, b_vec, arma::solve_opts::likely_sympd);
    if (!solved) {
      return 1e30;  // return large value if singular
    }

    // Quadratic form: quad = sum_d [ rss_d - c_d * rbar_d^2 ]
    double quad = 0.0;
    for (int d = 0; d < D; ++d) {
      double c_d = theta / (1.0 + stats[d].nd * theta);
      double rss_d = stats[d].S_yy
        - 2.0 * arma::dot(beta, stats[d].S_xy)
        + arma::as_scalar(beta.t() * stats[d].S_xx * beta);
      double rbar_d = stats[d].ybar - arma::dot(stats[d].xbar, beta);
      quad += rss_d - c_d * rbar_d * rbar_d;
    }

    // Profile sigma2_e
    double sigma2_e = quad / (n - p);
    if (sigma2_e <= 0) return 1e30;

    // Log determinants
    double log_det_V = n * std::log(sigma2_e) + log_det_V_part;

    // log|X'V^{-1}X| = log|A / sigma2_e| = log|A| - p*log(sigma2_e)
    double log_det_A_val;
    double log_det_A_sign;
    arma::log_det(log_det_A_val, log_det_A_sign, A);
    if (log_det_A_sign <= 0) return 1e30;
    double log_det_XVX = log_det_A_val - p * std::log(sigma2_e);

    // -2 * l_REML = (n-p)*log(2pi) + log|V| + log|X'V^{-1}X| + (n-p)
    double neg2_reml = (n - p) * std::log(2.0 * M_PI)
      + log_det_V + log_det_XVX + (n - p);

    return 0.5 * neg2_reml;
  };

  // 4. Optimize over log(theta) using Brent's method in [-15, 15]
  double best_log_theta = brent_minimize(neg_reml, -15.0, 15.0);

  return neg_reml(best_log_theta);
}
