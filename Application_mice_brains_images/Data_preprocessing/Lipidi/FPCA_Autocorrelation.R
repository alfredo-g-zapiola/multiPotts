
# libraries ---------------------------------------------------------------


knitr::opts_chunk$set(fig.width = 8,fig.height = 6)
library(tidyverse)
library(rayshader)
library(patchwork)
library(plyr)

# EDA ---------------------------------------------------------------------


D = read.table("101_lipidi-PreProcessed-IM-Step1-Step2-Step4-Step5-101.txt")
D0 = D
D0[is.na(D0)] = 0

pixels = read.table("101_lipidi-PreProcessed-XYCoordinates-Step1-Step2-Step4-Step5-101.txt")

colnames(D0) = substr(colnames(D0),1,4)
colnames(pixels) = c("x","y")
Data_long = as_tibble(data.frame( pixels, D0 ))
max_number_of_pixels = apply(Data_long[,1:2],2,max)

Data_array = matrix(NA,max_number_of_pixels[1],max_number_of_pixels[2])

Data_array = array(NA,c(max_number_of_pixels[1],max_number_of_pixels[2],ncol(D0)))

sum(is.na(D0))

head(Data_long)
for(k in 1:ncol(D0)){
  for(i in 1:nrow(Data_long)){
    Data_array[Data_long$x[i],Data_long$y[i],k] = D0[i,k]
  }
}

Data_very_long = reshape2::melt(Data_long,c("x","y")) %>% mutate(pixel_ind = paste0(x,"_",y), value_ind = rep(1:nrow(Data_long),ncol(D0)))

Data_very_long = Data_very_long %>% group_by(pixel_ind) %>% mutate(n = row_number()) %>% ungroup() %>% mutate(mz = as.numeric(substr(variable,2,4)))

# subsampling to get a faster plot and not drain memory
sub_ind = sample(unique(Data_very_long$pixel_ind),100)
# just to get the gist:
ggplot(Data_very_long %>% filter(pixel_ind %in% sub_ind))+
  geom_path(aes(x = mz, y = value, 
                col=pixel_ind, 
                group = pixel_ind),alpha=.5)+theme_bw()+theme(legend.position = "none")+xlab("m.z")+scale_color_viridis_d(option = "A")+
  scale_x_continuous(n.breaks = 20)

mz_values<-as.numeric(substr(Data_very_long$variable,2,4))
mz_values<-unique(mz_values)


# FPCA --------------------------------------------------------------------

library(fda)
install.packages("raster")
library(raster)    

basis <- create.bspline.basis(rangeval=c(401,967),breaks=mz_values[c(which(mz_values> 450 & mz_values < 550), which(mz_values>750))], norder = 2)
D1<-as.matrix(D0)
D1<-t(D1)
data_W.fd.1 <- Data2fd(argvals = mz_values,y = D1,basisobj = basis)
plot.fd(data_W.fd.1,xlab='mz',ylab='value')



pca_W.1 <- pca.fd(data_W.fd.1,nharm=5,centerfns=TRUE)

plot(pca_W.1$values[1:5],xlab='j',ylab='Eigenvalues')
plot(cumsum(pca_W.1$values)[1:5]/sum(pca_W.1$values),xlab='j',ylab='CPV',ylim=c(0.8,1))

# first two FPCs
layout(cbind(1,2))
plot(pca_W.1$harmonics[1,],col=1,ylab='FPC1')
abline(h=0,lty=2)
plot(pca_W.1$harmonics[2,],col=2,ylab='FPC2')


# plot of the FPCs as perturbation of the mean
media <- mean.fd(data_W.fd.1)

plot(media,lwd=2,ylab='value',main='FPC1')
lines(media+pca_W.1$harmonics[1,]*sqrt(pca_W.1$values[1]), col=2)
lines(media-pca_W.1$harmonics[1,]*sqrt(pca_W.1$values[1]), col=3)

plot(media,lwd=2,ylab='value',main='FPC2')
lines(media+pca_W.1$harmonics[2,]*sqrt(pca_W.1$values[2]), col=2)
lines(media-pca_W.1$harmonics[2,]*sqrt(pca_W.1$values[2]), col=3)

plot(media,lwd=2,ylab='value',main='FPC3')
lines(media+pca_W.1$harmonics[3,]*sqrt(pca_W.1$values[3]), col=2)
lines(media-pca_W.1$harmonics[3,]*sqrt(pca_W.1$values[3]), col=3)

plot(media,lwd=2,ylab='value',main='FPC4')
lines(media+pca_W.1$harmonics[4,]*sqrt(pca_W.1$values[4]), col=2)
lines(media-pca_W.1$harmonics[4,]*sqrt(pca_W.1$values[4]), col=3)

# scatter plot of the scores
plot(pca_W.1$scores[,1],pca_W.1$scores[,2],xlab="Scores FPC1",ylab="Scores FPC2",lwd=2)
library(ggplot2)

P1 = ggplot(Data_long)+ theme_bw()+
  geom_tile(aes(x=x,y=y,fill = Data_long[,3,drop = TRUE]))+scale_fill_viridis_c(option = "A",na.value = "red")+
  theme_void()+theme(legend.position = "bottom")

P2 = ggplot(Data_long)+
  geom_tile(aes(x=x,y=y,fill = X737))+scale_fill_viridis_c(option = "A",na.value = "red")+
  theme_void()+theme(legend.position = "bottom")

P3 = ggplot(Data_long)+
  geom_tile(aes(x=x,y=y,fill = X806))+scale_fill_viridis_c(option = "A",na.value = "red")+
  theme_void()+theme(legend.position = "bottom")
P4 = ggplot(Data_long)+
  geom_tile(aes(x=x,y=y,fill = X890))+scale_fill_viridis_c(option = "A",na.value = "red")+
  theme_void()+theme(legend.position = "bottom")

P1+P2+P3+P4

P1 = ggplot(Data_long)+ theme_bw()+
  geom_tile(aes(x=x,y=y,fill = pca_W.1$scores[,1]))+scale_fill_viridis_c(option = "A",na.value = "red")+
  theme_void()+theme(legend.position = "bottom")

P2 = ggplot(Data_long)+
  geom_tile(aes(x=x,y=y,fill = pca_W.1$scores[,2]))+scale_fill_viridis_c(option = "A",na.value = "red")+
  theme_void()+theme(legend.position = "bottom")

P3 = ggplot(Data_long)+
  geom_tile(aes(x=x,y=y,fill = pca_W.1$scores[,3]))+scale_fill_viridis_c(option = "A",na.value = "red")+
  theme_void()+theme(legend.position = "bottom")
P4 = ggplot(Data_long)+
  geom_tile(aes(x=x,y=y,fill = pca_W.1$scores[,4]))+scale_fill_viridis_c(option = "A",na.value = "red")+
  theme_void()+theme(legend.position = "bottom")

P1+P2+P3+P4

# Autocorrelation ---------------------------------------------------------

local_ac = matrix(NA,max_number_of_pixels[1],max_number_of_pixels[2])
num_functional_components = 4

for(i in seq(1,max_number_of_pixels[1],by = 1)){
  for(j in seq(1,max_number_of_pixels[2],by = 1)){
    if(!is.na(Data_array[i,j,1])){
      temp = data.frame(x = i, y = j)
      current_pixel_idx = as.integer(row.names(match_df(pixels, temp, on = TRUE)))
      local_ac[i,j] = 0
      n = 0;
      if(i < max_number_of_pixels[1] && !is.na(Data_array[i+1,j,1])){
        n = n+1
        temp = data.frame(x = i+1, y = j)
        neigh_pixel_idx = as.integer(row.names(match_df(pixels, temp, on = TRUE)))
        local_ac[i,j] = local_ac[i,j] + cor(pca_W.1$scores[current_pixel_idx,1:num_functional_components],pca_W.1$scores[neigh_pixel_idx,1:num_functional_components])
        
      }
      if(j < max_number_of_pixels[2] && !is.na(Data_array[i,j+1,1])){
        n = n+1
        temp = data.frame(x = i, y = j+1)
        neigh_pixel_idx = as.integer(row.names(match_df(pixels, temp, on = TRUE)))
        local_ac[i,j] = local_ac[i,j] + cor(pca_W.1$scores[current_pixel_idx,1:num_functional_components],pca_W.1$scores[neigh_pixel_idx,1:num_functional_components])
        
      }
      if( i > 1 && !is.na(Data_array[i-1,j,1])){
        n = n+1
        temp = data.frame(x = i-1, y = j)
        neigh_pixel_idx = as.integer(row.names(match_df(pixels, temp, on = TRUE)))
        local_ac[i,j] = local_ac[i,j] + cor(pca_W.1$scores[current_pixel_idx,1:num_functional_components],pca_W.1$scores[neigh_pixel_idx,1:num_functional_components])
        
        
      }
      if(j >1 && !is.na(Data_array[i,j-1,1])){
        n = n+1
        temp = data.frame(x = i, y = j - 1)
        neigh_pixel_idx = as.integer(row.names(match_df(pixels, temp, on = TRUE)))
        local_ac[i,j] = local_ac[i,j] + cor(pca_W.1$scores[current_pixel_idx,1:num_functional_components],pca_W.1$scores[neigh_pixel_idx,1:num_functional_components])
        
      }
      if(n != 0){
        local_ac[i,j] = local_ac[i,j]/n
      }
    }
  }
}


x11()
plot(raster(local_ac))
#bisogna fare in modo che le componenti contino il giusto proporzionalmente alla var spiegata
