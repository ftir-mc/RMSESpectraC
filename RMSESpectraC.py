#Python RMSE FTIR comparison software
print ("Softare for RMSE comparison")
print ("between FTIR spectra")
print ("implemented using sklearn")
print ("Fernando Gomes / LaBioS / IMA - UFRJ")
print ("V0.1 - 03/29/21")
print()
print()
print()
# Adapted from https://mubaris.com/posts/linear-regression/
# Importing Necessary Libraries
#%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

file = input("Insert your CSV file's name here (without .csv): ")
log = str(file)+'.log'
head_line = 'Samples ; RMSE ; R2'
print(head_line)
with open(log, 'a') as out:
        out.write(head_line + '\n')

# Reading Data
data = pd.read_csv(str(file)+'.csv')
#print(data.shape)
#data.head()

# Collecting X and Y
pos=[]
sample=[]
for i in range(0,(data.shape[1])):
    pos.append(i)
    sample.append(str(data.columns[pos[i]]))

colname = data.columns[pos]
X0 = data[colname[0]].values
X = data[colname[1]].values

for j in range(2, data.shape[1]):
    Y = data[colname[j]].values
    ##
    ### Total number of values
    m = len(X)
    
##    from sklearn.linear_model import LinearRegression
##    from sklearn.metrics import mean_squared_error

    # Cannot use Rank 1 matrix in scikit learn
    X = X.reshape((m, 1))
    # Creating Model
    reg = LinearRegression()
    # Fitting training data
    reg = reg.fit(X, Y)
    # Y Prediction
    Y_pred = reg.predict(X)

    # Calculating RMSE and R2 Score
    mse = mean_squared_error(Y, Y_pred)
    rmse = np.sqrt(mse)
    r2_score = reg.score(X, Y)

    print(sample[j],np.sqrt(mse),r2_score)
    var = str(sample[1])+ '_versus_'+ str(sample[j])+';'+str(np.sqrt(mse))+';'+str(r2_score)
    with open(log, 'a') as out:
        out.write(var + '\n')
    # Plotting Values and Regression Line
    plt.plot(X, Y_pred, color='#58b970', label='Linear fit')
    plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
    #plt.show()
    Xlabel=str(sample[1])+' - Nomalized Transmittance'
    Ylabel=str(sample[j])+' - Nomalized Transmittance'
    YlabelSpectrum=str(sample[j]+' - Nomalized Transmittance')
    plt.xlabel(Xlabel,size=22)
    plt.ylabel(Ylabel,size=22)
    plt.legend()
    plt.savefig(str(sample[1])+ '_versus_'+ str(sample[j])+'.png')
    plt.close()
    plt.xlabel('Wavenumber (cm-1)',size=22)
    plt.ylabel(YlabelSpectrum,size=22)
    plt.plot(X0, Y, color='#58b970', label='FTIR')
    plt.gca().invert_xaxis() # Tip from https://www.kite.com/python/answers/how-to-invert-the-y-axis-in-matplotlib-in-python
    plt.savefig('FTIR_'+ str(sample[j])+'.png')
    plt.close()
