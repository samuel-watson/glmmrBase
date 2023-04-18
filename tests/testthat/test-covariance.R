test_that("dense covariance tests", {
  df <- nelder(~(cl(4)*t(5)) > ind(5))
  df$int <- 0
  df[df$cl <= 2, 'int'] <- 1
  cptr <- expect_no_error(.Covariance__new("int+t+(1|gr(cl)*ar1(t))",as.matrix(df),colnames(df)))
  Z <- .Covariance__Z(cptr)
  expect_true(is(Z,"matrix"))
  expect_no_error(.Covariance__Update_parameters(cptr,c(0.25,0.8)))
  D <- .Covariance__D(cptr)
  expect_true(is(D,"matrix"))
  expect_equal(D[1,1],0.0625)
  L <- .Covariance__D_chol(cptr)
  expect_equal(L[1,1],0.25)
  expect_equal(L[5,2],0.0768)
  expect_equal(.Covariance__re_count(cptr),20)
  expect_equal(round(.Covariance__log_determinant(cptr),3),-71.798)
  re <- .Covariance__simulate_re(cptr)
  expect_true(is(re,"numeric"))
  expect_equal(length(re),20)
  expect_equal(round(.Covariance__log_likelihood(cptr,rep(0,20)),3),17.520)
  rm(cptr)
})

test_that("sparse covariance tests", {
  df <- nelder(~(cl(4)*t(5)) > ind(5))
  df$int <- 0
  df[df$cl <= 2, 'int'] <- 1
  cptr <- expect_no_error(.Covariance__new("int+t+(1|gr(cl)*ar1(t))",as.matrix(df),colnames(df)))
  expect_no_error(.Covariance__Update_parameters(cptr,c(0.25,0.8)))
  .Covariance__make_sparse(cptr)
  D <- .Covariance__D(cptr)
  expect_true(is(D,"matrix"))
  expect_equal(D[1,1],0.0625)
  L <- .Covariance__D_chol(cptr)
  expect_equal(L[1,1],0.25)
  expect_equal(L[5,2],0.0768)
  expect_equal(.Covariance__re_count(cptr),20)
  expect_equal(round(.Covariance__log_determinant(cptr),3),-71.798)
  re <- .Covariance__simulate_re(cptr)
  expect_true(is(re,"numeric"))
  expect_equal(length(re),20)
  expect_equal(round(.Covariance__log_likelihood(cptr,rep(0,20)),3),17.520)
  rm(cptr)
})

test_that("covariance functions", {
  df <- nelder(~(cl(4)*t(5)) > ind(5))
  cptr <- expect_no_error(.Covariance__new("(1|fexp0(t))",as.matrix(df),colnames(df)))
  .Covariance__Update_parameters(cptr,c(0.25))
  # add further checks of the other functions
  rm(cptr)
})
