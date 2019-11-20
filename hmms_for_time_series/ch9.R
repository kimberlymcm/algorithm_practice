library(MASS)
source('given_functions.R')


# Problem 4
# Consider the following bivariate normal-HMM on two states.
# In state 1, xt1 and xt2 are independently distributed as standard normal N(0,1)
# In state 2, they are independently distributed as N(5,1)

# a. Generate a realization of length 10,000 (say) from such an HMM
#    and plot the values of Xt1 against those of Xt2

gamma <- t(matrix(c(c(0.9, 0.1),
                    c(0.2, 0.8)), nrow=2))

mod$m <- NROW(gamma)
mod$delta <- statdist(gamma) # computes stationary dist from a transition matrix
mod$gamma <- gamma
mod$mean <- c(0, 5)
mod$variance <- c(1, 1)

HMM.generate_sample <- function(ns, mod) {
  mvect <- 1:mod$m # num states
  state <- numeric(ns) # num samples
  x <- rep(0, ns)
  state[1] <- sample(mvect, 1, prob=mod$delta)
  x[1] <- rnorm(1, mean=mod$mean[state[1]], sd=sqrt(mod$variance[state[1]]))
  for (i in 2:ns) {
    state[i] <- sample(mvect, 1, prob=mod$gamma[state[i-1],])
    x[i] <- rnorm(1, mean=mod$mean[state[i]], sd=sqrt(mod$variance[state[i]]))
  }
  result <- list()
  result$x <- x
  result$state <- state
  return(result)
}

result <- HMM.generate_sample(10000, mod)

plot(density(result$x[which(result$state == 1)]), col='red')
lines(density(result$x[which(result$state == 2)]), col='blue')


# b. Compute the (sample) correlation
acf(result$x)