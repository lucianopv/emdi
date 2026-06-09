#include <RcppArmadillo.h>
#include <string>
// [[Rcpp::depends(RcppArmadillo)]]

// Prasad-Rao-Singh analytical MSE for the spatial FH model.
// Faithful transcription of prasad_rao_spatial (R/mse.R:252-368). Dense.
// Returns the in-sample MSE vector (length m).
// method "reml" or "ml" (ml adds the -bML * grad(g1) correction).
// OOS-NA assembly is the R wrapper's job (Task 6).
// [[Rcpp::export]]
arma::vec fh_mse_spatial_cpp(double sigmau2, double rho,
                             const arma::mat& X, const arma::vec& vardir,
                             const arma::mat& W, const std::string& method) {
  const arma::uword m = X.n_rows;
  arma::mat I  = arma::eye(m, m);
  arma::mat Xt = X.t();
  arma::mat Wt = W.t();

  // DrhoWDrhoWt = solve((I - rho*Wt) %*% (I - rho*W))
  arma::mat A      = arma::inv((I - rho * Wt) * (I - rho * W));
  // var.DrhoW = sigmau2 * DrhoWDrhoWt
  arma::mat varG   = sigmau2 * A;
  // V.rho = var.DrhoW + diag(vardir)
  arma::mat Vr     = varG + arma::diagmat(vardir);
  // V.rhoi = solve(V.rho)
  arma::mat Vri    = arma::inv(Vr);
  // XtV.rhoi = Xt %*% V.rhoi
  arma::mat XtVri  = Xt * Vri;
  // Q.rho = solve(X' V.rhoi X)
  arma::mat Qr     = arma::inv(Xt * Vri * X);

  // G1 = var.DrhoW - var.DrhoW %*% V.rhoi %*% var.DrhoW
  arma::mat G1mat  = varG - varG * Vri * varG;
  // G2 = var.DrhoW %*% V.rhoi %*% X
  arma::mat G2mat  = varG * Vri * X;

  arma::vec g1(m), g2(m), g3(m), g4(m);

  // Computation of g1 and g2
  for (arma::uword d = 0; d < m; ++d) {
    g1(d) = G1mat(d, d);
    // Xcoef = X[d,] - G2[d,]
    arma::rowvec Xcoef = X.row(d) - G2mat.row(d);
    g2(d) = arma::as_scalar(Xcoef * Qr * Xcoef.t());
  }

  // der.rho = 2*rho * Wt %*% W - W - Wt
  arma::mat der_rho = 2.0 * rho * Wt * W - W - Wt;
  // DrhoWDrhoWtmat = -sigmau2 * (DrhoWDrhoWt %*% der.rho %*% DrhoWDrhoWt)
  arma::mat Amat    = -sigmau2 * (A * der_rho * A);
  // P = V.rhoi - t(XtV.rhoi) %*% Q.rho %*% XtV.rhoi
  arma::mat P       = Vri - XtVri.t() * Qr * XtVri;
  // PDrhoWDrhoWt = P %*% DrhoWDrhoWt
  arma::mat PA      = P * A;
  // PDrhoWDrhoWtmat = P %*% DrhoWDrhoWtmat
  arma::mat PAmat   = P * Amat;

  // Fisher information matrix (2x2)
  arma::mat fisher(2, 2);
  fisher(0, 0) = 0.5 * arma::trace(PA    * PA);
  fisher(0, 1) = 0.5 * arma::trace(PA    * PAmat);
  fisher(1, 0) = fisher(0, 1);
  fisher(1, 1) = 0.5 * arma::trace(PAmat * PAmat);
  arma::mat fisheri = arma::inv(fisher);

  // V.rhoiDrhoWDrhoWt    = V.rhoi %*% DrhoWDrhoWt
  arma::mat VriA    = Vri * A;
  // V.rhoiDrhoWDrhoWtmat = V.rhoi %*% DrhoWDrhoWtmat
  arma::mat VriAmat = Vri * Amat;

  // line1 = V.rhoiDrhoWDrhoWt - sigmau2 * V.rhoiDrhoWDrhoWt %*% V.rhoiDrhoWDrhoWt
  arma::mat line1  = VriA    - sigmau2 * VriA    * VriA;
  // line2 = V.rhoiDrhoWDrhoWtmat - sigmau2 * V.rhoiDrhoWDrhoWtmat %*% V.rhoiDrhoWDrhoWt
  arma::mat line2  = VriAmat - sigmau2 * VriAmat * VriA;
  arma::mat line1t = line1.t();
  arma::mat line2t = line2.t();

  // Computation of g3
  // lines (2 x m): lines[1,] = line1t[d,]; lines[2,] = line2t[d,]
  // g3[d] = trace(lines %*% V.rho %*% t(lines) %*% fisheri)
  for (arma::uword d = 0; d < m; ++d) {
    arma::mat lines(2, m);
    lines.row(0) = line1t.row(d);
    lines.row(1) = line2t.row(d);
    g3(d) = arma::trace(lines * Vr * lines.t() * fisheri);
  }

  arma::vec mse_help = g1 + g2 + 2.0 * g3;

  // Singh bias correction (g4)
  // psi = diag(vardir)
  arma::mat psi    = arma::diagmat(vardir);
  // D1help = -DrhoWDrhoWt %*% der.rho %*% DrhoWDrhoWt
  arma::mat D1help = -(A * der_rho * A);
  // D2help = 2*sigmau2 * DrhoWDrhoWt %*% der.rho %*% DrhoWDrhoWt %*% der.rho %*% DrhoWDrhoWt
  //        - 2*sigmau2 * DrhoWDrhoWt %*% Wt %*% W %*% DrhoWDrhoWt
  arma::mat D2help = 2.0 * sigmau2 * A * der_rho * A * der_rho * A
                   - 2.0 * sigmau2 * A * Wt * W * A;
  // D = (psi %*% V.rhoi %*% D1help %*% V.rhoi %*% psi) * (fisheri[1,2] + fisheri[2,1])
  //   + psi %*% V.rhoi %*% D2help %*% V.rhoi %*% psi * fisheri[2,2]
  arma::mat Dmat   = (psi * Vri * D1help * Vri * psi) * (fisheri(0, 1) + fisheri(1, 0))
                   + psi * Vri * D2help * Vri * psi * fisheri(1, 1);
  for (arma::uword d = 0; d < m; ++d) g4(d) = 0.5 * Dmat(d, d);

  arma::vec mse = mse_help - g4;

  if (method == "ml") {
    // Computation of bML
    // Q.rhoXtV.rhoi = Q.rho %*% XtV.rhoi
    arma::mat QrXtVri = Qr * XtVri;
    // V.rhoiX = V.rhoi %*% X
    arma::mat VriX    = Vri * X;
    // h1 = -trace(Q.rhoXtV.rhoi %*% DrhoWDrhoWt %*% V.rhoiX)
    double h1 = -arma::trace(QrXtVri * A    * VriX);
    // h2 = -trace(Q.rhoXtV.rhoi %*% DrhoWDrhoWtmat %*% V.rhoiX)
    double h2 = -arma::trace(QrXtVri * Amat * VriX);
    arma::vec h(2); h(0) = h1; h(1) = h2;
    // bML = (fisheri %*% h) / 2
    arma::vec bML = (fisheri * h) / 2.0;

    // Gradient of g1d
    // GV.rhoi = var.DrhoW %*% V.rhoi
    arma::mat GVri        = varG * Vri;
    // GV.rhoiDrhoWDrhoWt = GV.rhoi %*% DrhoWDrhoWt
    arma::mat GVriA       = GVri * A;
    // GV.rhoiDrhoWDrhoWtmat = GV.rhoi %*% DrhoWDrhoWtmat
    arma::mat GVriAmat    = GVri * Amat;
    // VriA already computed above (V.rhoi %*% DrhoWDrhoWt)
    // dg1_dDrhoWDrhoWt = DrhoWDrhoWt - 2*GVriA + sigmau2 * GVriA %*% VriA
    arma::mat dg1_dA      = A    - 2.0 * GVriA    + sigmau2 * GVriA    * VriA;
    // dg1_dp = DrhoWDrhoWtmat - 2*GVriAmat + sigmau2 * GVriAmat %*% VriA
    arma::mat dg1_dp      = Amat - 2.0 * GVriAmat + sigmau2 * GVriAmat * VriA;

    // mse[d] -= tbML %*% grad.g1d  (= bML[0]*dg1[0] + bML[1]*dg1[1])
    for (arma::uword d = 0; d < m; ++d) {
      arma::vec grad(2); grad(0) = dg1_dA(d, d); grad(1) = dg1_dp(d, d);
      mse(d) -= arma::as_scalar(bML.t() * grad);
    }
  }

  return mse;
}
