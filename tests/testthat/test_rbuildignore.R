# .Rbuildignore entries are regexes matched against relative paths, unanchored
# by default. An unescaped, unanchored ".git" therefore matches any path
# containing any character followed by "git" -- which silently excluded
# src/fh_logit.cpp from the built tarball ("l-o-g-i-t" contains "ogit").
#
# The symptom was maximally confusing: devtools::test() passed (it compiles the
# source tree directly), while R CMD check failed at install with
#   undefined symbol: fh_logit_integral_cpp(...)
# because RcppExports.cpp referenced a definition that never made it into the
# build.
#
# This pattern is inherited from upstream emdi and is still present in the
# released 2.2.3, so any file named *logit*, *digit* or *legitimate* would
# vanish from the tarball there too.

test_that(".Rbuildignore does not silently exclude source files", {
  ignore_path <- testthat::test_path("..", "..", ".Rbuildignore")
  skip_if_not(file.exists(ignore_path), "source tree not available")

  patterns <- Filter(nzchar, trimws(readLines(ignore_path)))

  # Every file that must reach the tarball.
  src <- c(list.files(testthat::test_path("..", "..", "src"),
                      pattern = "\\.(cpp|h)$"),
           list.files(testthat::test_path("..", "..", "R"), pattern = "\\.R$"))
  skip_if(length(src) == 0, "source tree not available")

  paths <- c(file.path("src", src), file.path("R", src))
  for (p in patterns) {
    hit <- paths[vapply(paths, function(f) grepl(p, f, perl = TRUE),
                        logical(1))]
    expect_length(hit, 0)
  }
})

test_that("the .git pattern is anchored", {
  ignore_path <- testthat::test_path("..", "..", ".Rbuildignore")
  skip_if_not(file.exists(ignore_path), "source tree not available")
  patterns <- trimws(readLines(ignore_path))

  expect_false("\\.git" %in% patterns)
  expect_false(".git" %in% patterns)     # the unanchored form that caused this
  expect_true(any(grepl("^\\^\\\\\\.git", patterns)))
})
