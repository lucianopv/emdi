# Test selected_domains parameter

# Load needed data
load("EBP/incomedata.RData")
load("EBP/Xoutsamp_AuxVar.RData")

test_that("selected_domains filters population data correctly", {
  suppressWarnings(RNGversion("3.5.0"))
  
  # Get all available domains
  all_domains <- unique(as.character(Xoutsamp_AuxVar$provlab))
  
  # Select only first 3 domains
  selected <- all_domains[1:3]
  
  # Run framework with selected_domains
  framework <- framework_ebp(
    income ~ educ1,
    Xoutsamp_AuxVar,
    "provlab",
    incomedata,
    "provlab",
    4282.081,
    custom_indicator = NULL,
    na.rm = TRUE,
    pop_weights = NULL,
    weights = NULL,
    selected_domains = selected
  )
  
  # Check that population data contains only selected domains
  expect_equal(length(unique(framework$pop_domains_vec)), 3)
  expect_true(all(unique(as.character(framework$pop_domains_vec)) %in% selected))
  
  # Check that sample data is unchanged
  expect_true(length(unique(framework$smp_domains_vec)) >= 3)
})

test_that("selected_domains works with ebp function", {
  suppressWarnings(RNGversion("3.5.0"))
  
  # Get all available domains
  all_domains <- unique(as.character(Xoutsamp_AuxVar$provlab))
  
  # Select only first 3 domains
  selected <- all_domains[1:3]
  
  set.seed(123)
  
  # Run ebp with selected_domains
  emdi_model <- ebp(
    fixed = income ~ educ1,
    pop_data = Xoutsamp_AuxVar,
    pop_domains = "provlab",
    smp_data = incomedata,
    smp_domains = "provlab",
    L = 2,
    na.rm = TRUE,
    selected_domains = selected
  )
  
  # Check that results contain only selected domains
  expect_equal(nrow(emdi_model$ind), 3)
  expect_true(all(as.character(emdi_model$ind$Domain) %in% selected))
})

test_that("selected_domains validation works", {
  suppressWarnings(RNGversion("3.5.0"))
  
  # Try with invalid domain names
  expect_error(
    ebp(
      fixed = income ~ educ1,
      pop_data = Xoutsamp_AuxVar,
      pop_domains = "provlab",
      smp_data = incomedata,
      smp_domains = "provlab",
      L = 2,
      na.rm = TRUE,
      selected_domains = c("InvalidDomain1", "InvalidDomain2")
    ),
    "selected_domains contains domain names that are not present"
  )
})

test_that("selected_domains NULL works as before", {
  suppressWarnings(RNGversion("3.5.0"))
  
  set.seed(123)
  
  # Run ebp without selected_domains
  emdi_model <- ebp(
    fixed = income ~ educ1,
    pop_data = Xoutsamp_AuxVar,
    pop_domains = "provlab",
    smp_data = incomedata,
    smp_domains = "provlab",
    L = 2,
    na.rm = TRUE,
    selected_domains = NULL
  )
  
  # Check that results contain all domains in population
  all_domains <- unique(as.character(Xoutsamp_AuxVar$provlab))
  expect_equal(nrow(emdi_model$ind), length(all_domains))
})
