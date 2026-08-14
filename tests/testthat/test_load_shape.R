# load_shapeaustria() reads the shape file that ships in inst/shapes/ via
# system.file(). system.file() returns "" rather than erroring when it cannot
# find the file, so a wrong package name there does not fail loudly -- load()
# is handed an empty path and dies with
#   cannot open compressed file '', probable reason 'No such file or directory'
# which says nothing about the real cause. R CMD check hit exactly that in
# ebp.Rd's --run-donttest examples.
#
# The failure was also maskable: on a machine where some other installed
# package happened to own a file at that path, system.file() would silently
# return that copy and the bug would stay invisible. So the first test pins the
# lookup to this package rather than trusting load_shapeaustria() to error.

test_that("the shape file ships in this package", {
  path <- system.file("shapes/shape_austria_dis.rda", package = "emdi")
  skip_if(!nzchar(path), "emdi not installed (devtools::load_all session)")
  expect_true(file.exists(path))
})

test_that("load_shapeaustria() loads a shape file", {
  skip_if_not_installed("sf")
  skip_if(!nzchar(system.file("shapes/shape_austria_dis.rda", package = "emdi")),
          "emdi not installed (devtools::load_all session)")
  on.exit(suppressWarnings(rm(list = "shape_austria_dis", envir = .GlobalEnv)),
          add = TRUE)

  expect_no_error(load_shapeaustria())
  expect_true(exists("shape_austria_dis", envir = .GlobalEnv))
})
