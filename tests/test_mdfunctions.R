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

#m is the size pf the square we sample from
m = 140

k = 4
betacritic = log(1 + sqrt(k))
mask = matrix(1,m,m)
neigh <- getNeighbors(mask = mask, c(2,2,0,0))
block <- getBlocks(mask = mask, 2)

sw_res <- swNoData(betacritic, k, neigh, block, niter = 1000, random = TRUE)

sinth_clust <- matrix(NA,m,m)
for(i in 1:m){
  for(j in 1:m){
    sinth_clust[i,j] = which.max(sw_res$z[m*(i-1)+j,])
  }
}

x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(sinth_clust, border=NA,asp = TRUE,col = viridis(k),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)

t = table(sinth_clust)

n = m*m
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

for(i in 1:m){
  for(j in 1:m){
    if(sinth_clust[i,j]==1){
      data[,m*(i-1)+j] = x1[c1,]
      c1 = c1+1
    }
    else if(sinth_clust[i,j]==2){
      data[,m*(i-1)+j] = x2[c2,]
      c2 = c2+1
    }
    else if (sinth_clust[i,j]==3){
      data[,m*(i-1)+j] = x3[c3,]
      c3 = c3+1
    }
    else {
      data[,m*(i-1)+j] = x4[c4,]
      c4 = c4+1
    }
  }
}

#random sampling inital mean
initmu1 = rmvnorm(1,mu1,sigma1)[1,]
initmu2 = rmvnorm(1,mu2,sigma2)[1,]
initmu3 = rmvnorm(1,mu3,sigma3)[1,]
initmu4 = rmvnorm(1,mu4,sigma4)[1,]

#this should come from an inverse wishart
initsigma1 = solve(rWishart(1,4+d,solve(sigma1)/(4+d))[,,1])
initsigma2 = solve(rWishart(1,4+d,solve(sigma2)/(4+d))[,,1])
initsigma3 = solve(rWishart(1,4+d,solve(sigma3)/(4+d))[,,1])
initsigma4 = solve(rWishart(1,4+d,solve(sigma4)/(4+d))[,,1])

prmu = cbind(mu1,mu2,mu3,mu4)
initmu = cbind(initmu1,initmu2,initmu3,initmu4)
initsigma = array(c(initsigma1,initsigma2,initsigma3,initsigma4),dim = c(d,d,k))
iter = 10000
burnin = 3000
r1 = diff(range(data[1,]))
r2 = diff(range(data[2,]))
M = matrix(c(r1^2,0,0,r2^2),2,2)
EV = matrix(c(var(data[1,]),cov(data[1,],data[2,]),cov(data[1,],data[2,]),var(data[2,])),2,2)

# gibbspotts

sourceCpp("C:/Users/Francesco/OneDrive - Politecnico di Milano/Bayesian Statistics Project/Codice/Cpp_code/GibbsSampler_updated.cpp")

priors <- list()
priors$k <- k
priors$mu <- prmu
priors$mu.sigma <- array(M,dim = c(d,d,k))
priors$sigma.V0 <- array(EV,dim = c(d,d,k))
priors$sigma.n0 <- rep(4+d,k)

system.time(results <- GibbsPotts(data,betacritic,initmu,initsigma,neigh,block,priors,iter,burnin))

res_clust = matrix(0,m,m)
for(i in 1:m){
  for(j in 1:m){
    res_clust[i,j] = which.max(results$alloc[m*(i-1)+j,])
  }
}

x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(res_clust, border=NA,asp = TRUE,col = viridis(k),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)

missclass = matrix(0,m,m)
diff = 0
for(i in 1:m){
  for(j in 1:m){
    if(!(res_clust[i,j]==sinth_clust[i,j])){
      diff = diff+1
      missclass[i,j] = 1
    }
  }
}

diff
error = diff/n
error

x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(missclass, border=NA,asp = TRUE,col = viridis(k),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)

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
for(i in seq(1,iter)){
  sigma111[i] = results$sigmas[i][[1]][1,1,1]
}
s1chain = mcmc(sigma111)
plot(s1chain)



# mcmcpotts


sourceCpp("C:/Users/Francesco/OneDrive - Politecnico di Milano/Bayesian Statistics Project/Codice/Cpp_code/mcmcPotts.cpp")

priorsmcmc <- list()
priorsmcmc$k <- k
priorsmcmc$mu <- prmu
priorsmcmc$mu.sigma <- array(M,dim = c(d,d,k))
priorsmcmc$sigma.V0 <- array(EV,dim = c(d,d,k))
priorsmcmc$sigma.n0 <- rep(4+d,k)
priorsmcmc$beta <- c(0,betacritic)

mh <- list(bandwidth=1,init = 1)

# the problem is that the mus get some NAN and therefor get excluded
#chek gibbslabels and gibbsbeta 
system.time(resmcmc <- MCMCPotts(data,neigh,block,iter,burnin,priorsmcmc,mh))

res_clustmcmc = matrix(0,m,m)
for(i in 1:m){
  for(j in 1:m){
    res_clustmcmc[i,j] = which.max(resmcmc$alloc[m*(i-1)+j,])
  }
}

x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(res_clustmcmc, border=NA,asp = TRUE,col = viridis(k),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)

res_clustmcmc2 = res_clustmcmc
for(i in 1:m){
  for(j in 1:m){
    
    if(res_clustmcmc[i,j]==4)
      res_clustmcmc2[i,j] = 1
    if(res_clustmcmc[i,j]==1)
      res_clustmcmc2[i,j] = 2
    if(res_clustmcmc[i,j]==2)
      res_clustmcmc2[i,j] = 4
    
  }
}

x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(res_clustmcmc2, border=NA,asp = TRUE,col = viridis(k),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)


diff = 0
for(i in 1:m){
  for(j in 1:m){
    if(!(res_clustmcmc2[i,j]==sinth_clust[i,j]))
      diff = diff+1
  }
}

diff
error = diff/n
error

mu1chain = mcmc(t(resmcmc$mu[,1,]))
mu2chain = mcmc(t(resmcmc$mu[,2,]))
mu3chain = mcmc(t(resmcmc$mu[,3,]))
mu4chain = mcmc(t(resmcmc$mu[,4,]))

x11()
plot(mu1chain)
autocorr.plot(mu1chain)
plot(mu2chain)
autocorr.plot(mu2chain)
plot(mu3chain)
autocorr.plot(mu3chain)
plot(mu4chain)
autocorr.plot(mu4chain)


#GMM test
sourceCpp("C:/Users/Francesco/OneDrive - Politecnico di Milano/Bayesian Statistics Project/Codice/Cpp_code/GibbsGMM.cpp")

priorsGMM <- list()
priorsGMM$k <- k
priorsGMM$mu <- prmu
priorsGMM$mu.sigma <- array(M,dim = c(d,d,k))
priorsGMM$sigma.V0 <- array(EV,dim = c(d,d,k))
priorsGMM$sigma.n0 <- rep(4+d,k)
priorsGMM$lambda <- rep(1,k)

system.time(resGMM <- gibbsGMMmd(data,iter,burnin,priorsGMM))

res_clustgmm = matrix(0,m,m)
for(i in 1:m){
  for(j in 1:m){
    res_clustgmm[i,j] = which.max(resGMM$alloc[m*(i-1)+j,])
  }
}

x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(res_clustgmm, border=NA,asp = TRUE,col = viridis(k),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)


res_clustgmm2 = res_clustgmm
for(i in 1:m){
  for(j in 1:m){
    if(res_clustgmm[i,j]==3)
      res_clustgmm2[i,j] = 1
    if(res_clustgmm[i,j]==2)
      res_clustgmm2[i,j] = 3
    if(res_clustgmm[i,j]==1)
      res_clustgmm2[i,j] = 2
  }
}

x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(res_clustgmm2, border=NA,asp = TRUE,col = viridis(k),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)


diff = 0
for(i in 1:m){
  for(j in 1:m){
    if(!(res_clustgmm2[i,j]==sinth_clust[i,j]))
      diff = diff+1
  }
}

diff
error = diff/n
error

mu1chain = mcmc(t(resGMM$mu[,1,]),start = burnin)
mu2chain = mcmc(t(resGMM$mu[,2,]),start = burnin)
mu3chain = mcmc(t(resGMM$mu[,3,]),start = burnin)
mu4chain = mcmc(t(resGMM$mu[,4,]),start = burnin)

x11()
plot(mu1chain)
autocorr.plot(mu1chain)
plot(mu2chain)
autocorr.plot(mu2chain)
plot(mu3chain)
autocorr.plot(mu3chain)
plot(mu4chain)
autocorr.plot(mu4chain)

sourceCpp("C:/Users/Francesco/OneDrive - Politecnico di Milano/Bayesian Statistics Project/Codice/Cpp_code/All_in_one.cpp")
