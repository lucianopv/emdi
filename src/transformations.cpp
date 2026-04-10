#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// ---------------------------------------------------------------------------
// back_transform_arma: Internal version returning arma::vec (no copy).
// Used by other C++ code to avoid Rcpp::NumericVector conversion overhead.
// ---------------------------------------------------------------------------
arma::vec back_transform_arma(const arma::vec& y,
                               const std::string& transformation,
                               double lambda,
                               double shift) {
  arma::vec out(y.n_elem);

  if (transformation == "no") {
    out = y;
  } else if (transformation == "log") {
    out = arma::exp(y) - shift;
  } else if (transformation == "box.cox") {
    if (std::abs(lambda) <= 1e-12) {
      out = arma::exp(y) - shift;
    } else {
      out = arma::pow(lambda * y + 1.0, 1.0 / lambda) - shift;
    }
  } else if (transformation == "dual") {
    if (std::abs(lambda) <= 1e-12) {
      out = arma::exp(y) - shift;
    } else {
      double lam2 = lambda * lambda;
      out = arma::pow(lambda * y + arma::sqrt(lam2 * (y % y) + 1.0),
                      1.0 / lambda) - shift;
    }
  } else if (transformation == "log.shift") {
    out = arma::exp(y) - lambda;
  }

  return out;
}

// Back-transform a vector y given the transformation type.
// Matches the R function back_transformation() in R/transformation_functions.R
//
// Transformation types:
//   "no"        -> y (identity)
//   "log"       -> exp(y) - shift
//   "box.cox"   -> (lambda*y + 1)^(1/lambda) - shift  [lambda!=0]
//                   exp(y) - shift                     [lambda~0]
//   "dual"      -> (lambda*y + sqrt(lambda^2*y^2+1))^(1/lambda) - shift [lambda!=0]
//                   exp(y) - shift                     [lambda~0]
//   "log.shift" -> exp(y) - lambda
//
// [[Rcpp::export]]
Rcpp::NumericVector back_transform_cpp(const arma::vec& y,
                                       const std::string& transformation,
                                       double lambda,
                                       double shift) {
  int n = y.n_elem;
  arma::vec out(n);

  if (transformation == "no") {
    out = y;
  } else if (transformation == "log") {
    out = arma::exp(y) - shift;
  } else if (transformation == "box.cox") {
    if (std::abs(lambda) <= 1e-12) {
      out = arma::exp(y) - shift;
    } else {
      out = arma::pow(lambda * y + 1.0, 1.0 / lambda) - shift;
    }
  } else if (transformation == "dual") {
    if (std::abs(lambda) <= 1e-12) {
      out = arma::exp(y) - shift;
    } else {
      double lam2 = lambda * lambda;
      out = arma::pow(lambda * y + arma::sqrt(lam2 * (y % y) + 1.0),
                      1.0 / lambda) - shift;
    }
  } else if (transformation == "log.shift") {
    out = arma::exp(y) - lambda;
  } else {
    Rcpp::stop("Unknown transformation type: " + transformation);
  }

  return Rcpp::NumericVector(out.begin(), out.end());
}
