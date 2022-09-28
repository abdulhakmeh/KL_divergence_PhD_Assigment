## KL divergance with closed form Gaussian distribuation

##This immplementaion of KL divergancebased on closed form Gaussian distribuation to calculate the density of two subsequances. 
No garantie of resut.
This work is done in 10 dayes as a PhD. assigment. 

## to run the code you need to install the follwing python packegs: 
sklearn, 
 pandas ,
 matplotlib ,
 numpy ,
 
make sure that the folder raw_data is located in your root dir. when you run the code. alternative you can change the path as it is passed as variable.the source file .py contains the source code. No need to pass parameters in case of run in console. after the run is finished  
the results will be shown as plt plot and saved locally in CSV format. for the sack of simplicity a _jupeterBook version is also provided and contains the same code as py file.

## dataset
The Excel file "HPW_2012_41046.xls"  in raw_data  contains four columns: 
Date, wave height (Hs), sea level pressure (P), and wind speed (W). 
The time series of the three marine variables (Hs, P, W) span six 
months of hourly data. There are several short multivariate 
intervals,  each of a maximum length of 7 days,  that represent 
anomalous events.
