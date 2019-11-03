setwd("/Users/kmcmanus/Documents/classes/algorithm_practice/hmms_for_time_series/")

library(MASS)
source('given_functions.R')

# Chapter 3 Question 2
# The purpose of this exercise is to investigate the numerical behaviour of an 'unscaled'
# evaluation of the likelihood of an HMM, and to compare this with the behaviour of an
# alternative algortihm that applies scaling.

gamma <- t(matrix(c(c(0.9, 0.1), c(0.2, 0.8)), nrow=2))
lambda <- c(1, 5)
observs <- c(2, 8, 6, 3, 6, 1, 0, 0, 4, 7)

# Unscaled method
delta <- statdist(gamma)
# P(state 1) * P(observation | state 1)
alpha <- delta * dpois(observs[1], lambda)
# P(state 1) * P(move to state 2 | state 1) * P(observation | state 2)
for (i in 2:length(observs)) {
  print(alpha)
  alpha <- alpha %*% P * dpois(observs[i], lambda)
}
sum(alpha) # 3.829457e-11

# Scaled
alpha <- delta * dpois(observs[1], lambda) # same
lscale <- log(sum(alpha))
alpha <- alpha/sum(alpha)
for (i in 2:length(observs)) {
  alpha <- alpha %*% gamma * dpois(observs[i], lambda)
  lscale <- lscale + log(sum(alpha))
  alpha <- alpha/sum(alpha) # Keep re-scaling alpha to sum to 1.
}
lscale

# Chapter 3, Question 10
# Consider again the soap sales series introduced in Exercise 6 of Chapter 1
# 10a: Fit stationary Poisson-HMMs with 2, 3, and 4 states to these data.

# Why does this give different answers than just fitting the poison mixture?
# (Because we make a time dependence assumption with the HMM)
# Overall the ll are better with in these models than in Chapter 1 Exercise 6,
# indicating that there is some time dependence.

# Two states
mod$gamma0 <- t(matrix(c(c(0.8, 0.2), c(0.2, 0.8)), nrow=2))
mod$m <- NROW(mod$gamma0)
mod$delta <- c(0.85, 0.15) # estimated in chr1
mod$lambda0 <- c(4.2, 12.6) # estimated in chr1
result_two <- pois.HMM.mle(soap_data, mod$m, mod$lambda0, mod$gamma0, delta0=NULL)

# Three states
mod$gamma0 <- t(matrix(c(c(0.8, 0.1, 0.1),
                         c(0.1, 0.8, 0.1),
                         c(0.1, 0.1, 0.8)), nrow=3))
mod$m <- NROW(mod$gamma0)
mod$delta <- c(0.77, 0.11, 0.12) # estimated in the code above
mod$lambda0 <- c(4.8, 1.43, 13.5) # estimated in the code above
result <- pois.HMM.mle(soap_data, mod$m, mod$lambda0, mod$gamma0, delta0=NULL)

# Four states
mod$gamma0 <- t(matrix(c(c(0.7, 0.1, 0.1, 0.1),
                         c(0.1, 0.7, 0.1, 0.1),
                         c(0.1, 0.1, 0.7, 0.1),
                         c(0.1, 0.1, 0.1, 0.7)), nrow=4))
mod$m <- NROW(mod$gamma0)
mod$delta <- c(0.02, 0.40, 0.48, 0.11) # estimated in the code above
mod$lambda0 <- c(4.801420e-06, 3.3, 5.6, 13.8) # estimated in the code above
result <- pois.HMM.mle(soap_data, mod$m, mod$lambda0, mod$gamma0, delta0=NULL)

# Ignore code below this for right now
# Implement Baum-Welch for Poisson Earthquake data
x <- c(13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18, 25, 21, 21, 14,
       8, 11, 14, 23, 18, 17, 19, 20, 22, 19, 13, 26, 13, 14, 22, 24, 21, 22, 26, 21,
       23, 24, 27, 41, 31, 27, 35, 26, 28, 36, 39, 21, 17, 22, 17, 19, 15, 34, 10, 15,
       22, 18, 15, 20, 15, 22, 19, 16, 30, 27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15,
       18, 14, 10, 15, 8, 15, 6, 11, 8, 7, 18, 16, 13, 12, 13, 20, 15, 16, 12, 18,
       15, 16, 13, 15, 16, 11, 11)

gamma <- t(matrix(c(c(0.9, 0.1),c(0.1, 0.9)), nrow=2))

mod$m <- NROW(gamma)
mod$delta <- c(0.5, 0.5)
mod$gamma <- gamma
mod$lambda <- c(10, 30)

# Write code to fit the Baum-Welch

#library(HMMpa)
#test <- Baum_Welch_algorithm(x=x, m=2, delta=c(0.5, 0.5), gamma=gamma,
#                     distribution_class="pois",
#                     distribution_theta=list(lambda = c(10, 30)),
#                     BW_limit_accuracy=0.00001)

alpha <- pois.HMM.lforward(x, mod)
beta <- pois.HMM.backward(x, mod)
zeta <- matrix(c(0), ncol=m, nrow = size)  
zeta <- exp(alpha + beta - logL) 


# Write code to fit normal, binomial, exponential HMMs by EM




