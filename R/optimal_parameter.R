optimal_parameter <- function(generic_opt,
                              fixed,
                              smp_data,
                              smp_domains,
                              transformation,
                              interval,
                              control) {
  if (transformation != "no" &&
    transformation != "log") {
    # no lambda -> no estimation -> no optmimization

    if (transformation == "box.cox" && any(interval == "default")) {
      interval <- c(-1, 2)
    } else if (transformation == "dual" && any(interval == "default")) {
      interval <- c(0, 2)
    } else if (transformation == "log.shift" && any(interval == "default")) {
      # interval = c(min(smp_data[paste(fixed[2])]),
      # max(smp_data[paste(fixed[2])]))
      span <- range(smp_data[paste(fixed[2])])
      if ((span[1] + 1) <= 1) {
        lower <- abs(span[1]) + 1
      } else {
        lower <- 0
      }

      upper <- diff(span) / 2

      interval <- c(lower, upper)
    }

    if (.use_cpp()) {
      # Sort data by domain for C++ (requires contiguous domain blocks)
      smp_data_sorted <- smp_data[order(smp_data[[smp_domains]]), ]
      y <- as.numeric(smp_data_sorted[[as.character(fixed[[2]])]])
      X <- model.matrix(fixed, smp_data_sorted)
      # droplevels: if smp_domains carries unused factor levels (e.g. survey
      # region2 aligned to census levels for OOS-domain coverage in pop_data),
      # table() would emit zero counts. The C++ sufficient-stats loop then
      # tries X.rows(offset, offset - 1) and Armadillo throws Mat::rows().
      # We only drop levels in the local domain_factor used to build n_d;
      # smp_data_sorted itself is untouched so the model matrix above and any
      # downstream factor-level checks see the original factor unchanged.
      domain_factor <- droplevels(as.factor(smp_data_sorted[[smp_domains]]))
      domain_ids <- as.integer(domain_factor)
      n_d <- as.integer(table(domain_factor))

      optimal_parameter <- optimal_parameter_cpp(
        y = y, X = X,
        domain_ids = domain_ids, n_d = n_d,
        transformation = transformation,
        lower = interval[1], upper = interval[2]
      )
    } else {
      # Upstream's implementation. Restored so that engine = "r" runs the
      # reference lambda search rather than silently falling through to C++:
      # someone setting "r" to investigate a suspect lambda must actually get
      # the R optimiser, or the check is worthless.
      #
      # It costs almost nothing to carry -- generic_opt() never left this
      # package (plot.ebp.R calls it to draw the lambda profile), so the R
      # branch is one optimize() call over a function that has to exist anyway.
      # This is also why the two agree only to optimizer tolerance and not
      # exactly: same likelihood and same Brent implementation, but R profiles
      # via generic_opt()'s nlme::lme() refits while the kernel uses the
      # closed-form per-domain sufficient statistics.
      optimal_parameter <- optimize(generic_opt,
        fixed          = fixed,
        smp_data       = smp_data,
        smp_domains    = smp_domains,
        transformation = transformation,
        interval       = interval,
        control        = control,
        maximum        = FALSE
      )$minimum
    }
  } else {
    optimal_parameter <- NULL
  }

  return(optimal_parameter)
} # End optimal parameter


# Internal documentation -------------------------------------------------------

# Function generic_opt provides estimation method reml to specifiy
# the optimal parameter lambda. Here its important that lambda is the
# first argument because generic_opt is given to optimize. Otherwise,
# lambda is missing without default.

generic_opt <- function(lambda,
                        fixed,
                        smp_data,
                        smp_domains,
                        transformation,
                        control) {


  # Definition of optimization function for finding the optimal lambda
  # Preperation to easily implement further methods here
  optimization <- if (TRUE) {
    reml(
      fixed = fixed,
      smp_data = smp_data,
      smp_domains = smp_domains,
      transformation = transformation,
    lambda = lambda,
    control = control
  )
}
  return(optimization)
}



# REML method ------------------------------------------------------------------

reml <- function(fixed = fixed,
                 smp_data = smp_data,
                 smp_domains = smp_domains,
                 transformation = transformation,
                 lambda = lambda,
                 control = control) {
  sd_transformed_data <- std_data_transformation(
    fixed = fixed,
    smp_data = smp_data,
    transformation = transformation,
    lambda = lambda
  )

  ## browser()

  model_REML <- NULL
  tryCatch({model_REML <- lme(
    fixed = fixed,
    data = sd_transformed_data,
    random =
      as.formula(paste0(
        "~ 1 | as.factor(",
        smp_domains, ")"
      )),
    method = "REML",
    keep.data = FALSE,
    control = nlme::lmeControl(opt = "optim")
  )},
    message = function(e) {
      model_REML <<- e
    }
  #  silent = TRUE)
  )
  if (is.null(model_REML)) {
    ## stop(strwrap(prefix = " ", initial = "",
    ##              "The likelihood does not converge. One reason could be that
    ##              the interval for the estimation of an optimal transformation
    ##              parameter is not appropriate. Try another interval. See also
    ##              help(ebp)."))
    print(model_REML$message)
  } else {
    model_REML <- model_REML
  }


  log_likelihood <- -logLik(model_REML)

  return(log_likelihood)
}
