setwd("/Users/kmcmanus/Documents/classes/algorithm_practice/hmms_for_time_series/")

library(MASS)
source('given_functions.R')

# Chapter 5 Question 4
# Apply local and global decoding to a 3 state model for tye soap sale series
# introduced in exercise 6 of chapter 1, and compare the results to see how
# much the conclusions differ

# First need to estimate the transitions / emissions from chr 1
# Inferred in chr3_4.R
gamma <- t(matrix(c(c(0.54, 0.445, 0.0169),
                    c(0.1169, 0.864, 0.0187),
                    c(0.297, 1.95e-8, 0.70)), nrow=3))

mod <- list()
mod$m <- NROW(gamma)
mod$delta <- c(0.2199392, 0.7220542, 0.0580066) # estimated in the code above
mod$gamma <- gamma
mod$lambda <- c(8.443471, 3.736170, 14.926813) # estimated in the code above

# Global decoding
result <- pois.HMM.viterbi(soap_data, mod)

# Local decoding
result3 <- pois.HMM.state_probs(soap_data, mod)
result2 <- pois.HMM.local_decoding(soap_data, mod)
table(result, result2) # This is how much they differ

# (Not an official question from the book, but more practice)
# Apply local and global decoding to the earthquake data
# Didn't work great because the parameters haven't been trained.
gamma <- t(matrix(c(c(0.8, 0.1, 0.1),
                    c(0.1, 0.8, 0.1),
                    c(0.1, 0.1, 0.8)), nrow=3))
mod <- list()
mod$m <- NROW(gamma)
mod$delta <- c(0.33, 0.33, 0.33) # estimated in the code above
mod$gamma <- gamma
mod$lambda <- c(2, 10, 20) # estimated in the code above

# Global decoding
result <- pois.HMM.viterbi(earthquake_data, mod)

# Local decoding
result3 <- pois.HMM.state_probs(earthquake_data, mod)
result2 <- pois.HMM.local_decoding(earthquake_data, mod)
table(result, result2) # This is how much they differ



# Chapter 6 Question 4
# 4a: Using the same sequence of random numbers in each case, generate sequences
# of length 1000 from the Poisson-HMMs with 
gamma <- t(matrix(c(c(0.8, 0.1, 0.1),
                    c(0.1, 0.8, 0.1),
                    c(0.1, 0.1, 0.8)), nrow=3))
mod <- list()
mod$m <- NROW(gamma)
mod$gamma <- gamma
mod$lambda <- c(10, 20, 30) # estimated in the code above
mod$delta<-solve(t(diag(mod$m)-mod$gamma+1),rep(1,mod$m)) # just 0.33 for all

# First set
ns <- 1000
mvect <- 1:mod$m
state <- numeric(ns)
state[1] <- 1 # just set first state to 1
for (i in 2:ns) state[i] <- sample(mvect,1,prob=mod$gamma[state[i-1],])
first_set <- rpois(ns, lambda=mod$lambda[state])
second_set <- rpois(ns, lambda=c(15,20,25))

first_set_result <- pois.HMM.viterbi(first_set, mod)
first_set_result2 <- pois.HMM.local_decoding(first_set, mod)
first_set_result3 <- pois.HMM.state_probs(first_set, mod)
table(state, first_set_result) # This is how much they differ
table(state, first_set_result2)

# These lambdas mean that the results are too overlapping, and neither
# algorithm can tell them apart very well.

mod$lambda <- c(15,20,25) # estimated in the code above
sec_set_result <- pois.HMM.viterbi(second_set, mod)
sec_set_result2 <- pois.HMM.local_decoding(second_set, mod)
sec_set_result3 <- pois.HMM.state_probs(second_set, mod)
table(state, sec_set_result) # This is how much they differ
table(state, sec_set_result2) 
