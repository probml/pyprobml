#https://datascienceplus.com/generalized-additive-models/
  
#requiring the Package 
require(gam)

#ISLR package contains the 'Wage' Dataset
require(ISLR)
attach(Wage) #Mid-Atlantic Wage Data

?Wage # To search more on the dataset
?gam() # To search on the gam function 

gam1<-gam(wage~s(age,df=6)+s(year,df=6)+education ,data = Wage)
#in the above function s() is the shorthand for fitting smoothing splines 
#in gam() function

summary(gam1)

#Plotting the Model
par(mfrow=c(1,3)) #to partition the Plotting Window
plot(gam1,se = TRUE) 
#se stands for standard error Bands
