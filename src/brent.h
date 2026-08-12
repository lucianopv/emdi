#ifndef EMDI_BRENT_H
#define EMDI_BRENT_H

#include <cmath>
#include <limits>

// Brent's method for one-dimensional minimisation.
//
// This is a faithful transcription of R's Brent_fmin
// (src/library/stats/src/optimize.c), which backs stats::optimize(). R does not
// export Brent_fmin from its public C API -- R_ext/Applic.h exposes only the
// multivariate optim backends (vmmin, nmmin, cgmin, lbfgsb, optif9) -- so a
// package that needs a 1-D minimiser has to carry its own.
//
// Transcribing R's version rather than using a library implementation (e.g.
// boost::math::tools::brent_find_minima) is deliberate: reproducing optimize()'s
// exact iteration path is what lets the C++ kernels be tested against R to
// optimizer tolerance. A different-but-equally-valid minimiser would land on a
// slightly different argmin and break that comparison.
//
// Templated on the callable so it serves both plain functions and lambdas;
// this replaced three separate copies of Brent that had accumulated in
// fh_reml.cpp, reml_optimization.cpp and lme_fit.cpp.

namespace emdi {

// R's optimize() default: .Machine$double.eps^0.25.
inline double brent_default_tol() {
  return std::pow(std::numeric_limits<double>::epsilon(), 0.25);
}

// Minimise f on [ax, bx]. `tol` is the additive term in R's convergence test
// (tol1 = sqrt(DBL_EPSILON) * |x| + tol/3), matching optimize()'s `tol`.
template <typename F>
double brent_fmin(double ax, double bx, F f, double tol) {
  const double c = (3.0 - std::sqrt(5.0)) * 0.5;
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  double a = ax, b = bx;
  double v = a + c * (b - a), w = v, x = v;
  double d = 0.0, e = 0.0;
  double fx = f(x), fv = fx, fw = fx;
  const double tol3 = tol / 3.0;

  for (;;) {
    double xm = (a + b) * 0.5;
    double tol1 = eps * std::fabs(x) + tol3;
    double t2 = 2.0 * tol1;
    if (std::fabs(x - xm) <= t2 - (b - a) * 0.5) break;

    double p = 0.0, q = 0.0, r = 0.0;
    if (std::fabs(e) > tol1) {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = (q - r) * 2.0;
      if (q > 0.0) p = -p; else q = -q;
      r = e; e = d;
    }
    if (std::fabs(p) >= std::fabs(0.5 * q * r) ||
        p <= q * (a - x) || p >= q * (b - x)) {
      e = (x < xm) ? (b - x) : (a - x);
      d = c * e;
    } else {
      d = p / q;
      double u_tmp = x + d;
      if (u_tmp - a < t2 || b - u_tmp < t2) {
        d = tol1;
        if (x >= xm) d = -d;
      }
    }
    double u;
    if (std::fabs(d) >= tol1) u = x + d;
    else if (d > 0.0)        u = x + tol1;
    else                     u = x - tol1;

    double fu = f(u);
    if (fu <= fx) {
      if (u < x) b = x; else a = x;
      v = w; fv = fw; w = x; fw = fx; x = u; fx = fu;
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) { v = w; fv = fw; w = u; fw = fu; }
      else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
    }
  }
  return x;
}

}  // namespace emdi

#endif  // EMDI_BRENT_H
