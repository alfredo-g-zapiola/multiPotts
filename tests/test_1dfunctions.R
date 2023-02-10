library(tidyverse)
library(plot.matrix)
library(bayesImageS)
library(viridis)
library(Rcpp)
library(RcppArmadillo)

D = read.table("/Users/macbookpro/Documents/Bayesian Statistics/Project/Raw_data/LIPIDI/78 variabili/101_lipidi-PreProcessed-IM-Step1-Step2-Step4-Step5-101.txt")
D0 = D
D0[is.na(D0)] = 0

pixels = read.table("/Users/macbookpro/Documents/Bayesian Statistics/Project/Raw_data/LIPIDI/78 variabili/101_lipidi-PreProcessed-XYCoordinates-Step1-Step2-Step4-Step5-101.txt")
colnames(D0) = substr(colnames(D0),1,4)
colnames(pixels) = c("x","y")

Data_long = as_tibble(data.frame( pixels, D0 ))
max_number_of_pixels = apply(Data_long[,1:2],2,max)

pca = princomp(D0)
pcascore1 <-as.vector(pca$scores[,1])

mask <-matrix(0,max_number_of_pixels[1],max_number_of_pixels[2])
for(i in 1:dim(pixels)[1]){
  mask[pixels[i,1],pixels[i,2]] = 1
}

neigh <- getNeighbors(mask = mask, c(2,2,0,0))
block <- getBlocks(mask = mask, 2)

q <- 5
betacritic = log(1 + sqrt(q))

mu <- c(-20,-10,0,7,18)
sd <- rep(2,q)

priors <- list()
priors$k <- q
priors$mu <- c(-20,-10,0,7,18)
priors$mu.sd <- rep(30,q)
priors$sigma <- rep(2,q)
priors$sigma.nu <- rep(0.125,q)

sourceCpp("/Users/macbookpro/Documents/Bayesian Statistics/Project/Cpp_code/GibbsSampler_updated.cpp")

resbc <- GibbsPotts(pcascore1, betacritic, mu, sd, neigh, block, priors, 10000,5000)

clusteringgibbs <-matrix(NA,max_number_of_pixels[1],max_number_of_pixels[2])
for(i in 1:dim(pixels)[1]){
  clusteringgibbs[pixels[i,1],pixels[i,2]] = which.max(resbc$alloc[i,])
}
x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(clusteringgibbs, border=NA,asp = TRUE,col = turbo(q),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)

muchain = mcmc(resbc$mu)
varnames(muchain)<-c("mu_1","mu_2","mu_3","mu_4","mu_5")
sigmachain = mcmc(resbc$sigma)
varnames(sigmachain)<-c("sigma_1","sigma_2","sigma_3","sigma_4","sigma_5")
sumchain  = mcmc(resbc$sum)
varnames(sumchain)<-c("sum")

summary(muchain)
batchSE(muchain)
effectiveSize(muchain)
rejectionRate(muchain)
plot(muchain)
par(mfrow=c(2,3))
autocorr.plot(muchain,auto.layout=FALSE)


sourceCpp("/Users/macbookpro/Documents/Bayesian Statistics/Project/Cpp_code/GibbsGMM.cpp")

priorsGMM <- list()
priorsGMM$k <- q
priorsGMM$lambda <- rep(1,q)
priorsGMM$mu <- rep(0,q)
priorsGMM$mu.sd <- rep(30,q)
priorsGMM$sigma <- rep(2,q)
priorsGMM$sigma.nu <- rep(0.125,q)

resGMM <- GibbsGMM(pcascore1, 5000,2000,priorsGMM)

clusteringgmm <-matrix(NA,max_number_of_pixels[1],max_number_of_pixels[2])
for(i in 1:dim(pixels)[1]){
  clusteringgmm[pixels[i,1],pixels[i,2]] = which.max(resGMM$alloc[i,])
}
x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(clusteringgmm, border=NA,asp = TRUE,col = turbo(q),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)

muchain = mcmc(resGMM$mu)
varnames(muchain)<-c("mu_1","mu_2","mu_3","mu_4","mu_5")
sigmachain = mcmc(resGMM$sigma)
varnames(sigmachain)<-c("sigma_1","sigma_2","sigma_3","sigma_4","sigma_5")
sumchain  = mcmc(resGMM$lambda)
varnames(sumchain)<-c("lambda_1","lambda_2","lambda_3","lambda_4","lambda_5")

summary(muchain)
batchSE(muchain)
effectiveSize(muchain)
rejectionRate(muchain)
plot(muchain)
par(mfrow=c(2,3))
autocorr.plot(muchain,auto.layout=FALSE)


sourceCpp("/Users/macbookpro/Documents/Bayesian Statistics/Project/Cpp_code/mcmcPotts.cpp")

priors <- list()
priors$k <- q
priors$mu <- c(-20,-10,0,7,18)
priors$mu.sd <- rep(30,q)
priors$sigma <- rep(2,q)
priors$sigma.nu <- rep(0.125,q)
priors$beta <- c(0,betacritic)

mh <- list(bandwidth=1,init = 1)

resmcmc <- MCMCPotts(pcascore1, neigh, block, 10000, 5000, priors, mh)
              
clusteringmcmc <-matrix(NA,max_number_of_pixels[1],max_number_of_pixels[2])
for(i in 1:dim(pixels)[1]){
  clusteringmcmc[pixels[i,1],pixels[i,2]] = which.max(resmcmc$alloc[i,])
}
x11()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(clusteringmcmc, border=NA,asp = TRUE,col = turbo(q),axis.col=NULL, axis.row=NULL, xlab='', ylab='',key = NULL)

muchain = mcmc(resmcmc$mu)
varnames(muchain)<-c("mu_1","mu_2","mu_3","mu_4","mu_5")
sigmachain = mcmc(resmcmc$sigma)
varnames(sigmachain)<-c("sigma_1","sigma_2","sigma_3","sigma_4","sigma_5")
sumchain  = mcmc(resmcmc$sum)
varnames(sumchain)<-c("sum")
betachain  = mcmc(resmcmc$beta)
varnames(betachain)<-c("beta")

summary(muchain)
batchSE(muchain)
effectiveSize(muchain)
rejectionRate(muchain)
plot(muchain)
par(mfrow=c(2,3))
autocorr.plot(muchain,auto.layout=FALSE)

summary(betachain)
batchSE(betachain)
effectiveSize(betachain)
rejectionRate(betachain)
plot(betachain)
autocorr.plot(betachain)
