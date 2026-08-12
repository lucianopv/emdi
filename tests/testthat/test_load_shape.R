# load_shapeaustria() looked the shape file up under package = "emdi", the
# pre-fork name, while the file ships in emdi2's inst/shapes/. system.file()
# returns "" for a package that is not installed, so load() got an empty path:
#   cannot open compressed file '', probable reason 'No such file or directory'
# R CMD check hit this in ebp.Rd's --run-donttest examples.
#
# It only surfaced once three earlier blockers were cleared, and it was masked
# on any machine where the original emdi happened to be installed -- there
# system.file() would silently return emdi's own copy.

test_that("the shape file is found in emdi2, not the pre-fork emdi", {
  path <- system.file("shapes/shape_austria_dis.rda", package = "emdi2")
  skip_if(!nzchar(path), "emdi2 not installed (devtools::load_all session)")
  expect_true(file.exists(path))
})

test_that("load_shapeaustria() loads a shape file", {
  skip_if_not_installed("sf")
  skip_if(!nzchar(system.file("shapes/shape_austria_dis.rda", package = "emdi2")),
          "emdi2 not installed (devtools::load_all session)")
  on.exit(suppressWarnings(rm(list = "shape_austria_dis", envir = .GlobalEnv)),
          add = TRUE)

  expect_no_error(load_shapeaustria())
  expect_true(exists("shape_austria_dis", envir = .GlobalEnv))
})
