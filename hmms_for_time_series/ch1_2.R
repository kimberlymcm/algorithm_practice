setwd("/Users/kmcmanus/Documents/classes/algorithm_practice/hmms_for_time_series/")

library(MASS)
source('given_functions.R')

# Chapter 1: Problem 5
# Number of days in 1910-1912 on which there appeared, in the "Times of London",
# i death notices in respect of women aged 80 or over at death
x <- c(rep(0, 162), rep(1, 267), rep(2, 271), rep(3, 185), rep(4, 111),
       rep(5, 61), rep(6, 27), rep(7, 8), rep(8, 3), rep(9, 1))


# Function to compute -log(likelihood)
mllk <- function(wpar, x) {
  zzz <- w2n(wpar)
  # zzz$delta = probs of being in each state
  # zzz$lambda = mean of each dist
  # x = observed values
  # Gets outer product of x and zzz$lambda
  # and applies poison density, which gets the prob of the x
  # value given it poisson with mean lambda.
  return(-sum(log(outer(x, zzz$lambda, dpois) %*% zzz$delta)))
}

# Function to transform natural to working parameters
n2w <- function(lambda, delta) {
  # -1 is all but the first in the array
  log(c(lambda, delta[-1]/(1-sum(delta[-1]))))
}

# Function to transform working to natural parameters
w2n <- function(wpar){
  m <- (length(wpar) + 1) / 2 # num dists (just 2 in this case)
  lambda <- exp(wpar[1:m])
  delta <- exp(c(0, wpar[(m+1):(2*m-1)]))
  return(list(lambda=lambda, delta=delta/sum(delta)))
}

# Specify initial values
wpar <- n2w(c(2, 3), c(0.5, 0.5))

# 5a: Use nlm or optim to fit a mixture of two Poisson distributions
#     to these observations
result <- nlm(mllk, wpar, x)
w2n(result$estimate)


# 5b: Fit also a single Poisson distribution to these data.
#     Is a single Poisson distribution adequate as a model?
#     (Personal note: the mle of a Poisson is the mean value)
ll <- -sum(log(outer(x, mean(x), dpois)))

# 5c: Fit a mixture of 3 Poisson dists to these observations.
#     How many do you think is necessary?
#     (Personal note: I think two dists are necessary, because the ll stops improving after 2)
wpar <- n2w(c(1, 2, 3), c(0.333, 0.333, 0.333))
result <- nlm(mllk, wpar, x)
w2n(result$estimate)

wpar <- n2w(c(1, 2, 2.5, 3), c(0.25, 0.25, 0.25, 0.25))
result <- nlm(mllk, wpar, x)
w2n(result$estimate)

# Chapter 1 Question 6: Comsoder the series of weekly sales (in integer units) of a
# particular soap product in a supermarket. Fit Poisson mixture models with one, two
# three and four components. How many components do you think are necessary?
# Personal answer: The likelihood continues to improve with more components, but the
# main inflection point is between 1 and 2. A.I.C. says 3 components is best.

# One component
ll_one <- sum(log(outer(soap_data, mean(soap_data), dpois)))

# Two components
wpar <- n2w(c(4, 10), c(0.5, 0.5))
result <- nlm(mllk, wpar, soap_data)
w2n(result$estimate)
ll_two <- -result$minimum

# Three components
wpar <- n2w(c(1, 4, 10), c(0.333, 0.333, 0.333))
result <- nlm(mllk, wpar, soap_data)
w2n(result$estimate)
ll_three <- -result$minimum

# Four components
wpar <- n2w(c(1, 3, 6, 10), c(0.25, 0.25, 0.25, 0.25))
result <- nlm(mllk, wpar, soap_data)
w2n(result$estimate)
ll_four <- -result$minimum

print(c(ll_one, ll_two, ll_three, ll_four))


# Chapter 1 Question 16: Write an R function rMC(n, m, gamma, delta=NULL) that generates
# a series of length n from an m-state Markov chain with t.p.m. gamma. If the initial
# state distribution is given, then it should be used; otherwise the stationary
# distribution should be used as the initial distribution

m <- matrix()

statdist <- function(gamma) {
  # pi <- (gamma^100)[1,]
  # computes stationary dist of Markov chain with trans matrix gamma
  # stationary dist (pi): pi = pi*gamma
  r <- eigen(gamma) # Get the eigenvectors of P, note: R returns right eigenvectors
  rvec <- r$vectors
  # left eigenvectors are the inverse of the right eigenvectors: lvec=ginv(r$vectors)
  # The eigenvalues
  lam <- r$values
  # Two ways of checking the spectral decomposition:
  # Standard definition: rvec%*%diag(lam)%*%ginv(rvec)
  lvec <- ginv(r$vectors)
  pi <- lvec[1,]/sum(lvec[1,])
  return(pi)
}

gamma <- t(matrix(c(c(0.95, 0.05), c(0.3, 0.7)), nrow=2))

P <- t(matrix(c(c(0.5,0.4,0.1), c(0.3,0.4,0.3), c(0.2,0.3,0.5)), nrow=3))
t(P)^100

rMC <- function(n, m, gamma, delta=NULL){
  if (is.null(delta)) { delta <- statdist(gamma) }
  chain <- c()
  cur_state <- sample(seq(1, m), size=1, prob=delta)
  chain <- append(chain, cur_state)
  for (i in 2:n) {
    rel_row <- gamma[cur_state, ]
    cur_state <- sample(seq(1, m), size=1, prob=rel_row)
    chain <- append(chain, cur_state)
  }
  return(chain)
}

chain <- rMC(100, 2, gamma, delta=NULL)

# Chapter 2 Problem 1b
# Find Pr(X1=0, X2=2, X3=1)
P <- t(matrix(c(c(0.1, 0.9), c(0.4, 0.6)), nrow=2))
lambda <- c(1, 3)
observs <- c(0, 1, 2)
delta <- statdist(P)

# P(state 1) * P(observation | state 1)
alpha <- delta * dpois(observs[1], lambda)
# P(state 1) * P(move to state 2 | state 1) * P(observation | state 2)
alpha2 <- alpha %*% P * dpois(observs[2], lambda)
alpha3 <- alpha2 %*% P * dpois(observs[3], lambda)
sum(alpha3)


# Chapter 2 Problem 5
gamma <- t(matrix(c(c(0.990, 0.005, 0.005),
                    c(0.010, 0.980, 0.010),
                    c(0.015, 0.015, 0.970)), nrow=3))

mod$m <- NROW(gamma)
mod$delta <- statdist(gamma)
mod$gamma <- gamma
mod$lambda <- c(1, 10, 20)

HMM.generate_sample <- function(ns, mod) {
  mvect <- 1:mod$m
  state <- numeric(ns)
  x <- rep(0, ns)
  state[1] <- sample(mvect, 1, prob=mod$delta)
  x[1] <- rnorm(1, mean=1, sd=sqrt(mod$lambda[state[1]]))
  for (i in 2:ns) {
    state[i] <- sample(mvect, 1, prob=mod$gamma[state[i-1],])
    x[i] <- rnorm(1, mean=1, sd=sqrt(mod$lambda[state[i]]))
  }
  return(x)
}

result <- HMM.generate_sample(10000, mod)
acf(result)
acf(abs(result))
acf(result^2)
