test_that("model dense ml", {
  df <- nelder(~(j(10) * t(3)) > i(5))
  y <- ytest1
  df$t1 <- I(df$t==1)*1
  df$t2 <- I(df$t==2)*1
  df$t3 <- I(df$t==3)*1
  mptr <- Model__new(y, "t2 + t3 + (1|gr(j))",
                      as.matrix(df),
                      colnames(df),
                      "gaussian","identity")
  expect_error(Model__update_beta(mptr,c(0,0)))
  Model__update_beta(mptr,c(0.1,0.1,0.1))
  expect_error(Model__update_theta(mptr,c(0.25,0.25)))
  Model__update_theta(mptr,c(0.25))
  Model__set_var_par(mptr,1)
  Model__set_offset(mptr,rep(0,length(y)))
  expect_no_error(Model__mcmc_sample(mptr,250,250,10))
  Model__update_u(mptr,matrix(0,nrow=10,ncol=10))
  expect_no_error(Model__update_W(mptr))
  Model__ml_beta(mptr)
  expect_false(isTRUE(all.equal(Model__get_beta(mptr),c(0.1,0.1,0.1))))
  Model__ml_theta(mptr)
  expect_false(isTRUE(all.equal(Model__get_theta(mptr),c(0.25))))
  expect_no_error(Model__nr_beta(mptr))
})

test_that("model dense la", {
  df <- nelder(~(j(10) * t(3)) > i(5))
  y <- ytest1
  df$t1 <- I(df$t==1)*1
  df$t2 <- I(df$t==2)*1
  df$t3 <- I(df$t==3)*1
  mptr <- Model__new(y, "t2 + t3 + (1|gr(j))",
                      as.matrix(df),
                      colnames(df),
                      "gaussian","identity")
  Model__update_beta(mptr,c(0.1,0.1,0.1))
  Model__update_theta(mptr,c(0.25))
  Model__set_var_par(mptr,1)
  Model__set_offset(mptr,rep(0,length(y)))
  Model__laplace_ml_beta_u(mptr)
  expect_false(isTRUE(all.equal(Model__get_beta(mptr),c(0.1,0.1,0.1))))
  Model__laplace_ml_theta(mptr)
  expect_false(isTRUE(all.equal(Model__get_theta(mptr),c(0.25))))
  Model__laplace_ml_beta_theta(mptr)
  expect_false(isTRUE(all.equal(Model__get_beta(mptr),c(0.1,0.1,0.1))))
  expect_false(isTRUE(all.equal(Model__get_theta(mptr),c(0.25))))
  Model__update_beta(mptr,c(0.1,0.1,0.1))
  Model__laplace_nr_beta_u(mptr)
  expect_false(isTRUE(all.equal(Model__get_beta(mptr),c(0.1,0.1,0.1))))
})

test_that("model sparse ml", {
  df <- nelder(~(j(10) * t(3)) > i(5))
  y <- ytest1
  df$t1 <- I(df$t==1)*1
  df$t2 <- I(df$t==2)*1
  df$t3 <- I(df$t==3)*1
  mptr <- Model__new(y, "t2 + t3 + (1|gr(j))",
                      as.matrix(df),
                      colnames(df),
                      "gaussian","identity")
  expect_error(Model__update_beta(mptr,c(0,0)))
  Model__update_beta(mptr,c(0.1,0.1,0.1))
  expect_error(Model__update_theta(mptr,c(0.25,0.25)))
  Model__update_theta(mptr,c(0.25))
  Model__set_var_par(mptr,1)
  Model__set_offset(mptr,rep(0,length(y)))
  Model__make_sparse(mptr)
  expect_no_error(Model__mcmc_sample(mptr,250,250,10))
  Model__update_u(mptr,matrix(0,nrow=10,ncol=10))
  expect_no_error(Model__update_W(mptr))
  Model__ml_beta(mptr)
  expect_false(isTRUE(all.equal(Model__get_beta(mptr),c(0.1,0.1,0.1))))
  Model__ml_theta(mptr)
  expect_false(isTRUE(all.equal(Model__get_theta(mptr),c(0.25))))
  expect_no_error(Model__nr_beta(mptr))
})

test_that("overall model class",{
  df <- nelder(~(j(5) * t(3)) > i(10))
  des <- expect_no_error(Model$new(
    covariance = list(
      formula =  ~(1|gr(j)*ar1(t)),
      parameters = c(0.25,0.7)
    ),
    mean = list(
      formula = ~factor(t)-1,
      parameters = c(0.1,0.2,0.3)
    ),
    data=df,
    family=gaussian()
  ))
  expect_s3_class(des,"Model")
  pwr <- des$power()
  expect_equal(round(pwr[3,3],2),0.38)
  rm(des)
})
