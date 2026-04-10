test_that("data_transform_cpp matches R data_transformation for all types", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))

  # log
  r_log <- log_transform(y, shift = 0)
  cpp_log <- data_transform_cpp(y, "log", 0)
  expect_equal(as.numeric(cpp_log$y), as.numeric(r_log$y), tolerance = 1e-10)
  expect_equal(cpp_log$shift, r_log$shift, tolerance = 1e-10)

  # box.cox
  for (lam in c(0.3, 0.7, 1.0)) {
    r_bc <- box_cox(y, lambda = lam, shift = 0)
    cpp_bc <- data_transform_cpp(y, "box.cox", lam)
    expect_equal(as.numeric(cpp_bc$y), as.numeric(r_bc$y), tolerance = 1e-10,
                 info = paste("box.cox lambda =", lam))
    expect_equal(cpp_bc$shift, r_bc$shift, tolerance = 1e-10)
  }

  # box.cox lambda ~ 0
  r_bc0 <- box_cox(y, lambda = 0, shift = 0)
  cpp_bc0 <- data_transform_cpp(y, "box.cox", 0)
  expect_equal(as.numeric(cpp_bc0$y), as.numeric(r_bc0$y), tolerance = 1e-10)

  # dual
  r_dual <- dual(y, lambda = 0.5, shift = 0)
  cpp_dual <- data_transform_cpp(y, "dual", 0.5)
  expect_equal(as.numeric(cpp_dual$y), as.numeric(r_dual$y), tolerance = 1e-10)

  # no
  cpp_no <- data_transform_cpp(y, "no", 0)
  expect_equal(as.numeric(cpp_no$y), y, tolerance = 1e-10)
})

test_that("data_transform_cpp handles negative values with shift", {
  y <- c(-5, -2, 0, 3, 10, 50)
  r_bc <- box_cox(y, lambda = 0.5, shift = 0)
  cpp_bc <- data_transform_cpp(y, "box.cox", 0.5)
  expect_equal(as.numeric(cpp_bc$y), as.numeric(r_bc$y), tolerance = 1e-10)
  expect_equal(cpp_bc$shift, r_bc$shift, tolerance = 1e-10)
})
