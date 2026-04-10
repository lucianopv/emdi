test_that("std_transform_y_cpp matches R box_cox_std", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))
  for (lam in c(0.2, 0.5, 0.7, 1.0, 1.5, -0.5)) {
    r_result <- box_cox_std(y, lam)
    cpp_result <- std_transform_y_cpp(y, "box.cox", lam)
    expect_equal(as.numeric(cpp_result), as.numeric(r_result),
                 tolerance = 1e-10, info = paste("box.cox lambda =", lam))
  }
  # lambda ~ 0
  r_result_0 <- box_cox_std(y, 0)
  cpp_result_0 <- std_transform_y_cpp(y, "box.cox", 0)
  expect_equal(as.numeric(cpp_result_0), as.numeric(r_result_0),
               tolerance = 1e-10, info = "box.cox lambda = 0")
})

test_that("std_transform_y_cpp matches R dual_std", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))
  for (lam in c(0.2, 0.5, 0.7, 1.0, 1.5)) {
    r_result <- dual_std(y, lam)
    cpp_result <- std_transform_y_cpp(y, "dual", lam)
    expect_equal(as.numeric(cpp_result), as.numeric(r_result),
                 tolerance = 1e-10, info = paste("dual lambda =", lam))
  }
  r_result_0 <- dual_std(y, 0)
  cpp_result_0 <- std_transform_y_cpp(y, "dual", 0)
  expect_equal(as.numeric(cpp_result_0), as.numeric(r_result_0),
               tolerance = 1e-10, info = "dual lambda = 0")
})

test_that("std_transform_y_cpp matches R log_shift_opt_std", {
  set.seed(42)
  y <- abs(rnorm(500, 20000, 8000))
  for (lam in c(100, 500, 1000, 5000)) {
    r_result <- log_shift_opt_std(y, lam)
    cpp_result <- std_transform_y_cpp(y, "log.shift", lam)
    expect_equal(as.numeric(cpp_result), as.numeric(r_result),
                 tolerance = 1e-10, info = paste("log.shift lambda =", lam))
  }
})

test_that("std_transform_y_cpp handles negative values with shift", {
  y <- c(-5, -2, 0, 3, 10, 50)
  r_result <- box_cox_std(y, 0.5)
  cpp_result <- std_transform_y_cpp(y, "box.cox", 0.5)
  expect_equal(as.numeric(cpp_result), as.numeric(r_result), tolerance = 1e-10)
})
