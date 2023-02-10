# test multtidimensional potts model with k = 2
library(tidyverse)
library(plot.matrix)
library(bayesImageS)
library(viridis)
library(Rcpp)
library(RcppArmadillo)
library(mvtnorm)
library(coda)

#generation of the test data  4 clusters of 2 dim data 
#we use the sw alg for the spatial location of the clusters

k = 4
betacritic = log(1 + sqrt(k))
mask = matrix(1,20,20)
neigh <- getNeighbors(mask = mask, c(2,2,0,0))
block <- getBlocks(mask = mask, 2)

sw_res <- swNoData(betacritic, k, neigh, block, niter = 1000, random = TRUE)

sinth_clust <- matrix(NA,20,20)
for(i in 1:20){
  for(j in 1:20){
    sinth_clust[i,j] = which.max(sw_res$z[(i-1)*20+j,])
  }
}

x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(sinth_clust, border=NA,asp = TRUE,col = viridis(k),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)

t = table(sinth_clust)

n = 400
d = 2

n1 = t[1]
n2 = t[2]
n3 = t[3]
n4 = t[4]

mu1 = c(3,3)
mu2 = c(5,2)
mu3 = c(9,4)
mu4 = c(8,6)

sigma1 = matrix(c(1,0.5,0.5,1),2,2)
sigma2 = matrix(c(0.4,-0.2,-0.2,0.4),2,2)
sigma3 = matrix(c(1,0,0,1),2,2)
sigma4 = matrix(c(0.1,0,0,1),2,2)

x1 = rmvnorm(n1,mu1,sigma1)
x2 = rmvnorm(n2,mu2,sigma2)
x3 = rmvnorm(n3,mu3,sigma3)
x4 = rmvnorm(n4,mu4,sigma4)

data = matrix(0,d,n)

c1 = 1
c2 = 1
c3 = 1
c4 = 1

for(i in 1:20){
  for(j in 1:20){
    if(sinth_clust[i,j]==1){
      data[,20*(i-1)+j] = x1[c1,]
      c1 = c1+1
    }
    else if(sinth_clust[i,j]==2){
      data[,20*(i-1)+j] = x2[c2,]
      c2 = c2+1
    }
    else if (sinth_clust[i,j]==3){
      data[,20*(i-1)+j] = x3[c3,]
      c3 = c3+1
    }
    else {
      data[,20*(i-1)+j] = x4[c4,]
      c4 = c4+1
    }
  }
}


mu = cbind(mu1,mu2,mu3,mu4)
sigma = array(c(sigma1,sigma2,sigma3,sigma4),dim = c(d,d,k))
iter = 1000
r1 = diff(range(data[1,]))
r2 = diff(range(data[2,]))
M = matrix(c(r1^2,0,0,r2^2),2,2)
EV = matrix(c(var(data[1,]),cov(data[1,],data[2,]),cov(data[1,],data[2,]),var(data[2,])),2,2)

priors <- list()
priors$k <- k
priors$mu <- mu
priors$mu.sigma <- array(M,dim = c(d,d,k))
priors$sigma.V0 <- array(EV,dim = c(d,d,k))
priors$sigma.n0 <- rep(4+d,k)

# loading the functions
sourceCpp("/Users/macbookpro/Documents/Bayesian Statistics/Project/Cpp_code/GibbsSampler.cpp")
# prototype
# SEXP mdgibbsPotts(SEXP yS, SEXP betaS, SEXP muS, SEXP sigmaS, SEXP nS, SEXP bS, SEXP prS, SEXP itS)

results <- mdgibbsPotts(data,betacritic,mu,sigma,neigh,block,priors,iter)

res_clust = matrix(0,20,20)
for(i in 1:20){
  for(j in 1:20){
    res_clust[i,j] = which.max(results$z[20*(i-1)+j,])
  }
}

x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(res_clust, border=NA,asp = TRUE,col = viridis(k),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)

mu1chain = mcmc(t(results$mu[,1,]))
mu2chain = mcmc(t(results$mu[,2,]))
mu3chain = mcmc(t(results$mu[,3,]))
mu4chain = mcmc(t(results$mu[,4,]))

x11()
plot(mu1chain)
autocorr.plot(mu1chain)
plot(mu2chain)
autocorr.plot(mu2chain)
plot(mu3chain)
autocorr.plot(mu3chain)
plot(mu4chain)
autocorr.plot(mu4chain)

sigma111 = rep(0,iter)
for(i in range(iter)){
  sigma111[i] = results$lambdas[[i]][1,1,1]
}
s1chain = mcmc(sigma111)
plot(s1chain)
