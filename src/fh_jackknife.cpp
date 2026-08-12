#include <RcppArmadillo.h>
#include <string>
// [[Rcpp::depends(RcppArmadillo)]]

#ifdef _OPENMP
  #include <omp.h>
#endif

// From fh_reml.cpp
double fh_estsigmau2_reml_cpp(const arma::vec& direct, const arma::mat& X,
                              const arma::vec& vardir,
                              double lower, double upper, double tol);

// ---------------------------------------------------------------------------
// Jiang jackknife MSE for the standard Fay-Herriot model.
//
// For each in-sample domain j, the R implementation (jiang_jackknife in
// R/mse.R) re-estimates sigmau2 on the data with domain j deleted, then
// evaluates g1 and the EBLUP on the FULL in-sample data at that sigmau2:
//
//   g1[d]      = vardir[d] * sigmau2   / (sigmau2   + vardir[d])
//   g1_j[d]    = vardir[d] * sigmau2_j / (sigmau2_j + vardir[d])
//   FH_j       = X beta_hat(sigmau2_j) + u_hat(sigmau2_j)
//   mse = g1 - (m-1)/m * sum_j (g1_j - g1) + (m-1)/m * sum_j (FH_j - FH)^2
//
// Doing that in R costs, per domain, a framework_FH() rebuild (which runs
// makeXY/model.matrix), a wrapper_estsigmau2() dispatch and an eblup_FH()
// round trip. Here the whole loop runs on (direct, X, vardir).
//
// The loop is RNG-free and every iteration is independent, so unlike the
// bootstrap kernels it parallelises without any pre-generation of random
// numbers and is exactly reproducible regardless of thread count.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List fh_jackknife_cpp(const arma::vec& direct,
                            const arma::mat& X,
                            const arma::vec& vardir,
                            double sigmau2,
                            const arma::vec& fh_full,
                            double lower, double upper, double tol,
                            int threads = 1) {
  const arma::uword m = direct.n_elem;
  const arma::uword p = X.n_cols;

  // All argument validation happens here, before the parallel region: the R
  // API (Rcpp::stop) must not be touched from an OpenMP worker thread.
  if (X.n_rows != m) Rcpp::stop("fh_jackknife_cpp: nrow(X) != length(direct)");
  if (vardir.n_elem != m) Rcpp::stop("fh_jackknife_cpp: length(vardir) != length(direct)");
  if (fh_full.n_elem != m) Rcpp::stop("fh_jackknife_cpp: length(fh_full) != length(direct)");
  if (lower < 0.0) Rcpp::stop("fh_jackknife_cpp: lower must be >= 0");
  if (m < 3) Rcpp::stop("fh_jackknife_cpp: need at least 3 in-sample domains");

  if (threads < 1) threads = 1;

  const arma::vec g1 = vardir % (sigmau2 / (sigmau2 + vardir));

  arma::vec sum_dg1(m, arma::fill::zeros);   // sum_j (g1_j - g1)
  arma::vec sum_deb2(m, arma::fill::zeros);  // sum_j (FH_j - FH)^2
  arma::vec jack_sigmau2(m, arma::fill::zeros);

  bool failed = false;
  std::string errmsg;

  #ifdef _OPENMP
  #pragma omp parallel num_threads(threads)
  #endif
  {
    // Thread-private accumulators, combined once at the end. This keeps memory
    // flat in m rather than materialising the m x m difference matrices the R
    // version builds as data frames.
    arma::vec loc_dg1(m, arma::fill::zeros);
    arma::vec loc_deb2(m, arma::fill::zeros);

    #ifdef _OPENMP
    #pragma omp for schedule(static) nowait
    #endif
    for (long j = 0; j < (long) m; ++j) {
      if (failed) continue;
      try {
        // Leave-one-out subset.
        arma::uvec keep(m - 1);
        { arma::uword k = 0;
          for (arma::uword i = 0; i < m; ++i) if ((long) i != j) keep(k++) = i; }

        arma::vec direct_j = direct.elem(keep);
        arma::mat X_j      = X.rows(keep);
        arma::vec vardir_j = vardir.elem(keep);

        double s2j = fh_estsigmau2_reml_cpp(direct_j, X_j, vardir_j,
                                            lower, upper, tol);
        jack_sigmau2(j) = s2j;

        // g1 at the jackknifed sigmau2, on the full set of domains.
        loc_dg1 += vardir % (s2j / (s2j + vardir)) - g1;

        // EBLUP at s2j on the full data (mirrors eblup_FH called with
        // framework_insample, which is the complete in-sample framework).
        arma::vec vi = 1.0 / (s2j + vardir);
        arma::mat XtViX(p, p, arma::fill::zeros);
        arma::vec XtViy(p, arma::fill::zeros);
        for (arma::uword i = 0; i < m; ++i) {
          arma::rowvec xi = X.row(i);
          XtViX += vi(i) * (xi.t() * xi);
          XtViy += vi(i) * xi.t() * direct(i);
        }
        arma::vec beta_j = arma::inv_sympd(XtViX) * XtViy;
        arma::vec fh_j   = X * beta_j + s2j * (vi % (direct - X * beta_j));

        arma::vec d = fh_j - fh_full;
        loc_deb2 += d % d;
      } catch (std::exception& e) {
        #ifdef _OPENMP
        #pragma omp critical (fh_jk_err)
        #endif
        { failed = true; errmsg = e.what(); }
      } catch (...) {
        #ifdef _OPENMP
        #pragma omp critical (fh_jk_err)
        #endif
        { failed = true; errmsg = "unknown error"; }
      }
    }

    #ifdef _OPENMP
    #pragma omp critical (fh_jk_reduce)
    #endif
    {
      sum_dg1  += loc_dg1;
      sum_deb2 += loc_deb2;
    }
  }

  if (failed) Rcpp::stop("fh_jackknife_cpp: " + errmsg);

  const double cc = (double)(m - 1) / (double) m;
  arma::vec mse = g1 - cc * sum_dg1 + cc * sum_deb2;

  return Rcpp::List::create(
    Rcpp::Named("mse")          = mse,
    Rcpp::Named("jack_sigmau2") = jack_sigmau2
  );
}
