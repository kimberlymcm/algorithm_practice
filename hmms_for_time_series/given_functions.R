
# Soap data
soap_data <- c(1, 6, 9, 18, 14, 8, 8, 1, 6, 7, 3, 3, 1, 3, 4, 12, 8, 10, 8, 2,
               17, 15, 7, 12, 22, 10, 4, 7, 5, 0, 2, 5, 3, 4, 4, 7,  5, 6, 1, 3,
               4, 5, 3, 7, 3, 0, 4, 5, 3, 3, 4, 4, 4, 4, 4, 3, 5, 5, 5, 7,
               4, 0, 4, 3, 2, 6, 3, 8, 9, 6, 3, 4, 3, 3, 3, 3, 2, 1, 4, 5,
               5, 2, 7, 5, 2, 3, 1, 3, 4, 6, 8, 8, 5, 7, 2, 4, 2, 7, 4, 15,
               15, 12, 21, 20, 13, 9, 8, 0, 13, 9, 8, 0, 6, 2, 0, 3, 2, 4, 4, 6,
               3, 2, 5, 5, 3, 2, 1, 1, 3, 1, 2, 6, 2, 7, 3, 2, 4, 1, 5, 6,
               8, 14, 5, 3, 6, 5, 11, 4, 5, 9, 9, 7, 9, 8, 3, 4, 8, 6, 3, 5,
               6, 3, 1, 7, 4, 9, 2, 6, 6, 4, 6, 6, 13, 7, 4, 8, 6, 4, 4, 4,
               9, 2, 9, 2, 2, 2, 13, 13, 4, 5, 1, 4, 6, 5, 4, 2, 3, 10, 6, 15,
               5, 9, 9, 7, 4, 4, 2, 4, 2, 3, 8, 15, 0, 0, 3, 4, 3, 4, 7, 5,
               7, 6, 0, 6, 4, 14, 5, 1, 6, 5, 5, 4, 9, 4, 14, 2, 2, 1, 5, 2,
               6, 4)

# Earthquake data
earthquake_data <- c(13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18, 25, 21, 21, 14,
                     8, 11, 14, 23, 18, 17, 19, 20, 22, 19, 13, 26, 13, 14, 22, 24, 21, 22, 26, 21,
                     23, 24, 27, 41, 31, 27, 35, 26, 28, 36, 39, 21, 17, 22, 17, 19, 15, 34, 10, 15,
                     22, 18, 15, 20, 15, 22, 19, 16, 30, 27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15,
                     18, 14, 10, 15, 8, 15, 6, 11, 8, 7, 18, 16, 13, 12, 13, 20, 15, 16, 12, 18,
                     15, 16, 13, 15, 16, 11, 11)

# NOTE: Below functions mostly provided in book
# A.1.1 Transforming natural parameters to working
pois.HMM.pn2pw <- function(m,lambda,gamma,delta=NULL,stationary=TRUE)
{
  tlambda <- log(lambda)
  if(m==1) return(tlambda)
  foo     <- log(gamma/diag(gamma))
  tgamma  <- as.vector(foo[!diag(m)])
  if(stationary) {tdelta  <- NULL}
  else {tdelta <- log(delta[-1]/delta[1])}
  parvect <- c(tlambda,tgamma,tdelta)
  return(parvect)
}

# A.1.2 Transforming working parameters to natural
pois.HMM.pw2pn <- function(m,parvect,stationary=TRUE)
{
  lambda        <- exp(parvect[1:m])
  gamma         <- diag(m)
  if (m==1) return(list(lambda=lambda,gamma=gamma,delta=1))
  gamma[!gamma] <- exp(parvect[(m+1):(m*m)])
  gamma         <- gamma/apply(gamma,1,sum)
  if(stationary){delta<-solve(t(diag(m)-gamma+1),rep(1,m))}
  else {foo<-c(1,exp(parvect[(m*m+1):(m*m+m-1)]))
  delta<-foo/sum(foo)}
  return(list(lambda=lambda,gamma=gamma,delta=delta))
}



# A.1.3 Computing minus the log-likelihood from the working parameters
pois.HMM.mllk <- function(parvect,x,m,stationary=TRUE,...)
{
  if(m==1) return(-sum(dpois(x,exp(parvect),log=TRUE)))
  n        <- length(x)
  pn       <- pois.HMM.pw2pn(m,parvect,stationary=stationary)
  foo      <- pn$delta*dpois(x[1],pn$lambda)
  sumfoo   <- sum(foo)
  lscale   <- log(sumfoo)
  foo      <- foo/sumfoo
  for (i in 2:n)
  {
    if(!is.na(x[i])){P<-dpois(x[i],pn$lambda)}
    else {P<-rep(1,m)}
    foo    <- foo %*% pn$gamma*P
    sumfoo <- sum(foo)
    lscale <- lscale+log(sumfoo)
    foo    <- foo/sumfoo
  }
  mllk     <- -lscale
  return(mllk)
}


# A.1.4 Computing the MLEs, given starting values for the natural parameters
pois.HMM.mle <-
  function(x,m,lambda0,gamma0,delta0=NULL,stationary=TRUE,...)
  {
    parvect0  <- pois.HMM.pn2pw(m,lambda0,gamma0,delta0,stationary=stationary) # parameter conversion
    mod       <- nlm(pois.HMM.mllk,parvect0,x=x,m=m,stationary=stationary) # maximizing likelihood
    pn        <- pois.HMM.pw2pn(m=m,mod$estimate,stationary=stationary) #parameter conversion
    mllk      <- mod$minimum # best likelihood
    np        <- length(parvect0)
    AIC       <- 2*(mllk+np)
    n         <- sum(!is.na(x))
    BIC       <- 2*mllk+np*log(n)
    list(m=m,lambda=pn$lambda,gamma=pn$gamma,delta=pn$delta,code=mod$code,mllk=mllk,AIC=AIC,BIC=BIC)
  }


# A.1.5 Generating a sample
pois.HMM.generate_sample  <- function(ns,mod)
{
  mvect                    <- 1:mod$m
  state                    <- numeric(ns)
  state[1]                 <- sample(mvect,1,prob=mod$delta)
  for (i in 2:ns) state[i] <- sample(mvect,1,prob=mod$gamma[state[i-1],])
  x                        <- rpois(ns,lambda=mod$lambda[state])
  return(x)
}


# Global decoding - Viterbi algorithm
# Faster because it doesn't use information from the future and
# doesn't calculate every possibility, just the most likely one
pois.HMM.viterbi <- function(x, mod) {
  n <- length(x)
  xi <- matrix(0, n, mod$m) # # 0 matrix with num_observs x num_states
  foo <- mod$delta * dpois(x[1], mod$lambda) # P(hidden state i) * P(observed_state | each_hidden_state)
  xi[1, ] <- foo / sum(foo) # normalize
  for (i in 2:n) {
    # # most likely state to transition to at time i  * emissions
    foo <- apply(xi[i-1, ] * mod$gamma, 2, max) * dpois(x[i], mod$lambda)
    xi[i, ] <- foo / sum(foo)
  }
  iv <- numeric(n)
  iv[n] <- which.max(xi[n, ]) # get max at each state
  for (i in (n-1):1) {
    iv[i] <- which.max(mod$gamma[, iv[i+1]] * xi[i, ])
  }
  return(iv)
}

# A.1.7 Computing log(forward probabilities)
pois.HMM.lforward<-function(x,mod)
{
  n             <- length(x)
  lalpha        <- matrix(NA,mod$m,n)
  foo           <- mod$delta*dpois(x[1],mod$lambda)
  sumfoo        <- sum(foo)
  lscale        <- log(sumfoo)
  foo           <- foo/sumfoo
  lalpha[,1]    <- lscale+log(foo)
  for (i in 2:n)
  {
    foo          <- foo%*%mod$gamma*dpois(x[i],mod$lambda)
    sumfoo       <- sum(foo)
    lscale       <- lscale+log(sumfoo)
    foo          <- foo/sumfoo
    lalpha[,i]   <- log(foo)+lscale
  }
  return(lalpha)
}



# A.1.8 Computing log(backward probabilities)
pois.HMM.lbackward<-function(x,mod)
{
  n          <- length(x)
  m          <- mod$m
  lbeta      <- matrix(NA,m,n)
  lbeta[,n]  <- rep(0,m)
  foo        <- rep(1/m,m)
  lscale     <- log(m)
  for (i in (n-1):1)
  {
    foo        <- mod$gamma%*%(dpois(x[i+1],mod$lambda)*foo)
    lbeta[,i]  <- log(foo)+lscale
    sumfoo     <- sum(foo)
    foo        <- foo/sumfoo
    lscale     <- lscale+log(sumfoo)
  }
  return(lbeta)
}

# A.1.9 Conditional probabilities
# Conditional probability that observation at time t equals
# xc, given all observations other than that at time t.
# Note: xc is a vector and the result (dxc) is a matrix.
pois.HMM.conditional <- function(xc,x,mod)
{
  n         <- length(x)
  m         <- mod$m
  nxc       <- length(xc)
  dxc       <- matrix(NA,nrow=nxc,ncol=n)
  Px        <- matrix(NA,nrow=m,ncol=nxc)
  for (j in 1:nxc) Px[,j] <-dpois(xc[j],mod$lambda)
  la        <- pois.HMM.lforward(x,mod)
  lb        <- pois.HMM.lbackward(x,mod)
  la        <- cbind(log(mod$delta),la)
  lafact    <- apply(la,2,max)
  lbfact    <- apply(lb,2,max)
  for (i in 1:n)
  {
    foo      <-
      (exp(la[,i]-lafact[i])%*%mod$gamma)*exp(lb[,i]-lbfact[i])
    foo      <- foo/sum(foo)
    dxc[,i]  <- foo%*%Px
  }
  return(dxc)
}

# A.1.10 Pseudo-residuals
pois.HMM.pseudo_residuals <- function(x,mod)
{
  n        <- length(x)
  cdists   <- pois.HMM.conditional(xc=0:max(x),x,mod)
  cumdists <- rbind(rep(0,n),apply(cdists,2,cumsum))
  ulo <- uhi <- rep(NA,n)
  for (i in 1:n)
  {
    ulo[i]  <- cumdists[x[i]+1,i]
    uhi[i]  <- cumdists[x[i]+2,i]
  }
  umi       <- 0.5*(ulo+uhi)
  npsr     <- qnorm(rbind(ulo,umi,uhi))
  return(npsr)
}

# A.1.11 State probabilities
pois.HMM.state_probs <- function(x,mod)
{
  n          <- length(x) # num observations
  la         <- pois.HMM.lforward(x,mod) # forward_probs P(observed_state_at_t, C_t = j)
  lb         <- pois.HMM.lbackward(x,mod) # backward_probs, P(observed_state_at_t+1 | C_t = j)
  c          <- max(la[,n]) # final state
  llk        <- c+log(sum(exp(la[,n]-c)))
  stateprobs <- matrix(NA,ncol=n,nrow=mod$m)
  for (i in 1:n) stateprobs[,i]<-exp(la[,i]+lb[,i]-llk) # a*b / total_ll
  return(stateprobs)
}

# A.1.12 State prediction
# Note that state output 'statepreds' is a matrix even if h=1.
pois.HMM.state_prediction <- function(h=1,x,mod)
{
  n          <- length(x)
  la         <- pois.HMM.lforward(x,mod)
  c          <- max(la[,n])
  llk        <- c+log(sum(exp(la[,n]-c)))
  statepreds <- matrix(NA,ncol=h,nrow=mod$m)
  foo <- exp(la[,n]-llk)
  for (i in 1:h){
    foo<-foo%*%mod$gamma
    statepreds[,i]<-foo
  }
  return(statepreds)
}

### A.1.13 Local decoding
pois.HMM.local_decoding <- function(x,mod)
{
  n          <- length(x)
  stateprobs <- pois.HMM.state_probs(x,mod)
  ild        <- rep(NA,n)
  for (i in 1:n) ild[i]<-which.max(stateprobs[,i])
  ild
}

# A.1.14 Forecast probabilities
# Note that the output 'dxf' is a matrix.
pois.HMM.forecast <- function(xf,h=1,x,mod)
{
  n        <- length(x)
  nxf      <- length(xf)
  dxf      <- matrix(0,nrow=h,ncol=nxf)
  foo      <- mod$delta*dpois(x[1],mod$lambda)
  sumfoo   <- sum(foo)
  lscale   <- log(sumfoo)
  foo      <- foo/sumfoo
  for (i in 2:n)
  {
    foo    <- foo%*%mod$gamma*dpois(x[i],mod$lambda)
    sumfoo <- sum(foo)
    lscale <- lscale+log(sumfoo)
    foo    <- foo/sumfoo
  }
  for (i in 1:h)
  {
    foo    <- foo%*%mod$gamma
    for (j in 1:mod$m) dxf[i,] <- dxf[i,] + foo[j]*dpois(xf,mod$lambda[j])
  }
  return(dxf)
}