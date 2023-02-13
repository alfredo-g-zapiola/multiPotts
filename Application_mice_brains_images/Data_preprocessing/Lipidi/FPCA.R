library(tidyverse)
library(rayshader)
library(patchwork)
library(ggplot2)
library(fda)

# usual data loading steps

D = read.table("/Users/macbook/Documents/Bayesian Statistics/Project/Raw_data/LIPIDI/78 variabili/101_lipidi-PreProcessed-IM-Step1-Step2-Step4-Step5-101.txt")
D0 = D
D0[is.na(D0)] = 0

pixels = read.table("/Users/macbook/Documents/Bayesian Statistics/Project/Raw_data/LIPIDI/78 variabili/101_lipidi-PreProcessed-XYCoordinates-Step1-Step2-Step4-Step5-101.txt")
colnames(D0) = substr(colnames(D0),1,4)
colnames(pixels) = c("x","y")

Data_long = as_tibble(data.frame( pixels, D0 ))
max_number_of_pixels = apply(Data_long[,1:2],2,max)

Data_very_long = reshape2::melt(Data_long,c("x","y")) %>% mutate(pixel_ind = paste0(x,"_",y), value_ind = rep(1:nrow(Data_long),ncol(D0)))
Data_very_long = Data_very_long %>% group_by(pixel_ind) %>% mutate(n = row_number()) %>% ungroup() %>% mutate(mz = as.numeric(substr(variable,2,4)))

# CONVENTOIONAL PCA WITH B-SPLINES a preliminary approach

mz_values<-as.numeric(substr(Data_very_long$variable,2,4))
mz_values<-unique(mz_values)
length(mz_values)

basis <- create.bspline.basis(rangeval=c(401,967),nbasis=25)
D1<-as.matrix(D0)
D1<-t(D1)
data_W.fd.1 <- Data2fd(y = D1,argvals = mz_values,basisobj = basis)
plot.fd(data_W.fd.1,xlab='mz',ylab='value')

#FPCA
pca_W.1 <- pca.fd(data_W.fd.1,nharm=5,centerfns=TRUE)

plot(pca_W.1$values[1:78],xlab='j',ylab='Eigenvalues')
plot(cumsum(pca_W.1$values)[1:78]/sum(pca_W.1$values),xlab='j',ylab='CPV',ylim=c(0.8,1))

# first two FPCs
layout(cbind(1,2))
plot(pca_W.1$harmonics[1,],col=1,ylab='FPC1')
abline(h=0,lty=2)
plot(pca_W.1$harmonics[2,],col=2,ylab='FPC2')

# plot of the FPCs as perturbation of the mean
media <- mean.fd(data_W.fd.1)

plot(media,lwd=2,ylab='value',main='FPC1', ylim=c(0,10))
lines(media+pca_W.1$harmonics[1,]*sqrt(pca_W.1$values[1]), col=2)
lines(media-pca_W.1$harmonics[1,]*sqrt(pca_W.1$values[1]), col=3)

plot(media,lwd=2,ylab='value',main='FPC2', ylim=c(0,10))
lines(media+pca_W.1$harmonics[2,]*sqrt(pca_W.1$values[2]), col=2)
lines(media-pca_W.1$harmonics[2,]*sqrt(pca_W.1$values[2]), col=3)

# scatter plot of the scores
par(mfrow=c(1,1))
plot(pca_W.1$scores[,1],pca_W.1$scores[,2],xlab="Scores FPC1",ylab="Scores FPC2",lwd=2)

FPCA1 = ggplot(Data_long %>% mutate(pca1 = pca_W.1$scores[,1]))+ theme_bw()+
  geom_tile(aes(x=x,y=y,fill = pca1))+scale_fill_viridis_c(option = "A",na.value = "red")+
  theme_void()+theme(legend.position = "bottom")
FPCA1

# take the first two to get 99% of the variance
fpca_scores = pca_W.1$scores[,1:2] 

# experimenting with different basis

basis2 <- create.bspline.basis(rangeval=c(401,967),breaks = mz_values)
plot(basis2)

data_W.fd.2 <- Data2fd(y = D1,argvals = mz_values,basisobj = basis2)
plot.fd(data_W.fd.2,xlab='mz',ylab='value')

# specifying the knots is worse for uor case
rm(basis2)
rm(data_W.fd.2)
# experimenting with positive functions
basis <- create.bspline.basis(rangeval=c(401,967),nbasis=25)

Daux = D
Daux[is.na(Daux)] = exp(-20)
D1aux = t(Daux)


lam = 1e3
accleLfd = int2Lfd(2)
Wfdparobj = fdPar(fdobj =  basis,lambda = lam)
spectrpos = smooth.pos(mz_values,D1aux[,1],Wfdparobj)
Wfd = spectrpos$Wfdobj
plot((Lfd))

precfit = exp(eval.fd(mz_values,Lfd))

plot(mz_values,D1[,1])
lines(mz_values,precfit[,1])



