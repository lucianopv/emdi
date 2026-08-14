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
  path <- system.file("shapes/shape_austria_dis.rda", package = "emdi")
  # system.file() returns "" rather than erroring when it cannot find the file,
  # which would hand load() an empty path and fail with the unhelpful
  #   cannot open compressed file '', probable reason 'No such file or directory'
  # Name the actual problem instead.
  if (!nzchar(path)) {
    stop("Could not locate the Austrian districts shape file in the installed ",
         "emdi package.", call. = FALSE)
  }
  load(file = path, envir = .GlobalEnv)
}
