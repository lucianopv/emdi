#' Loading the Shape File for Austrian Districts
#'
#' The function simplifies to load the shape file for Austrian districts.
#'
#' @return A shape file of class \code{"sf", "data.frame"}.
#' @details The shape file contains the borders of Austrian districts. Thus, it
#' can be used for the visualization of estimation results for Austrian
#' districts.
#' @export

load_shapeaustria <- function() {
  # package = "emdi2", not "emdi". The shape file ships in this package's
  # inst/shapes/; looking it up under the pre-fork name made system.file()
  # return "" whenever emdi itself was not installed, so load() was handed an
  # empty path and failed with
  #   cannot open compressed file '', probable reason 'No such file or directory'
  # Missed in the emdi -> emdi2 rename, the same oversight as tests/testthat.R.
  path <- system.file("shapes/shape_austria_dis.rda", package = "emdi2")
  if (!nzchar(path)) {
    stop("Could not locate the Austrian districts shape file in the installed ",
         "emdi2 package.", call. = FALSE)
  }
  load(file = path, envir = .GlobalEnv)
}
