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
