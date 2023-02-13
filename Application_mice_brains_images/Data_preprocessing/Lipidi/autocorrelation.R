
library(plot.matrix)

# loading the data and some processing on the data stuctures

D = read.table("/Users/macbook/Documents/Bayesian Statistics/Project/Raw_data/LIPIDI/78 variabili/101_lipidi-PreProcessed-IM-Step1-Step2-Step4-Step5-101.txt")
sum(is.na(D))

# we replace the missing data with 0 since it means the data for that mz was under threshold

D0 = D
D0[is.na(D0)] = 0
pixels = read.table("/Users/macbook/Documents/Bayesian Statistics/Project/Raw_data/LIPIDI/78 variabili/101_lipidi-PreProcessed-XYCoordinates-Step1-Step2-Step4-Step5-101.txt")
colnames(D0) = substr(colnames(D0),1,4)
colnames(pixels) = c("x","y")

#Create the datasets we will need:
#  - `Data_long`: contains the 18k pixels on the rows. The first two columns are the coordinates, 
#the remaining 78 are the values recorded for different `m.z`.
#- `Data_array`: contains a cube with the 78 slices on the third dimension. Not all the pixels are recorded in each rectangle: beware of `NA`'s.

Data_long = as_tibble(data.frame( pixels, D0 ))
max_number_of_pixels = apply(Data_long[,1:2],2,max)
dim(Data_long)

Data_array = matrix(NA,max_number_of_pixels[1],max_number_of_pixels[2])
Data_array = array(NA,c(max_number_of_pixels[1],max_number_of_pixels[2],ncol(D0)))
for(k in 1:ncol(D0)){
  for(i in 1:nrow(Data_long)){
    Data_array[Data_long$x[i],Data_long$y[i],k] = D0[i,k]
  }
}

dim(Data_array)

Data_very_long = reshape2::melt(Data_long,c("x","y")) %>% mutate(pixel_ind = paste0(x,"_",y), value_ind = rep(1:nrow(Data_long),ncol(D0)))
Data_very_long = Data_very_long %>% group_by(pixel_ind) %>% mutate(n = row_number()) %>% ungroup() %>% mutate(mz = as.numeric(substr(variable,2,4)))
# data_very_long has the info in data_long but in a different manner to be uset for plotting

# I want to calculate spatial correlation of the data

#lets first start calculating the avg correlation in the tower neighbrood for each pixel

local_ac <- matrix(NA,max_number_of_pixels[1],max_number_of_pixels[2])

for(i in seq(1,max_number_of_pixels[1],by = 1)){
  for(j in seq(1,max_number_of_pixels[2],by = 1)){
    if(!is.na(Data_array[i,j,1])){
      local_ac[i,j] = 0
      n = 0;
      if(i < max_number_of_pixels[1] && !is.na(Data_array[i+1,j,1])){
        n = n+1
        local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i+1,j,])
        
      }
      if(j < max_number_of_pixels[2] && !is.na(Data_array[i,j+1,1])){
        n = n+1
        local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i,j+1,])
      }
      if( i > 1 && !is.na(Data_array[i-1,j,1])){
        n = n+1
        local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i-1,j,])
        
      }
      if(j >1 && !is.na(Data_array[i,j-1,1])){
        n = n+1
        local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i,j-1,])
      }
      if(n != 0){
        local_ac[i,j] = local_ac[i,j]/n
      }
    }
  }
}

par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(local_ac, border=NA,asp = TRUE,axis.col=NULL, axis.row=NULL, xlab='', ylab='')

# this justifies the use of the HRMF 
# the problems in the border are explainable with the problem of the instrument in that location

# doing it with a bigger neighborooh

local_ac <- matrix(NA,max_number_of_pixels[1],max_number_of_pixels[2])

for(i in seq(1,max_number_of_pixels[1],by = 1)){
  for(j in seq(1,max_number_of_pixels[2],by = 1)){
    if(!is.na(Data_array[i,j,1])){
      local_ac[i,j] = 0
      n = 0;
      if(i < max_number_of_pixels[1]){
        if(!is.na(Data_array[i+1,j,1])){
        n = n+1
        local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i+1,j,])
        }
        if(j < max_number_of_pixels[2] && !is.na(Data_array[i+1,j+1,1])){
          n = n+1
          local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i+1,j+1,])
        }
        if(j >1 && !is.na(Data_array[i+1,j-1,1])){
          n = n+1
          local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i+1,j-1,])
        }
      }
      if(i > 1){
        if(!is.na(Data_array[i-1,j,1])){
          n = n+1
          local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i-1,j,])
        }
        if(j < max_number_of_pixels[2] && !is.na(Data_array[i-1,j+1,1])){
          n = n+1
          local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i-1,j+1,])
        }
        if(j >1 && !is.na(Data_array[i-1,j-1,1])){
          n = n+1
          local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i-1,j-1,])
        }
          
      }
      if(j < max_number_of_pixels[2] && !is.na(Data_array[i,j+1,1])){
        n = n+1
        local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i,j+1,])
      }
      if(j >1 && !is.na(Data_array[i,j-1,1])){
        n = n+1
        local_ac[i,j] = local_ac[i,j] + cor(Data_array[i,j,],Data_array[i,j-1,])
      }
      if(n != 0){
        local_ac[i,j] = local_ac[i,j]/n
      }
    }
  }
}

par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(local_ac, border=NA,asp = TRUE,axis.col=NULL, axis.row=NULL, xlab='', ylab='')
