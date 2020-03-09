import os
import csv
import matplotlib.pyplot as plt
from numpy import exp, linspace, sqrt, diag
from scipy.optimize import curve_fit
import datetime

date = datetime.date.today()
yesterday = datetime.date.today() - datetime.timedelta(days=1)
bef_yesterday = datetime.date.today() - datetime.timedelta(days=2)

url = "https://raw.githubusercontent.com/DavideMagno/ItalianCovidData/master/Daily_Covis19_Italian_Data_Cumulative.csv"
os.system("wget -O data.csv "+url)

data = []

def sigmoid(x, a, b, c):
    return a/(1 + exp(-b*(x-c)))

def exponential(x, a, b):
    return a*exp(b*x)

with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first_row = True
    for row in csv_reader:
        if first_row:
            first_row = False
        else:
            data.append(int(row[2])+int(row[4]))

delta_t = 30
delta_t2 = 10
x = range(int(len(data)/21))
x2 = range(int(len(data)/21)+delta_t)
t = linspace(0,int(len(data)/21)+delta_t,100)
data = [sum(data[i*21:i*21+21]) for i in x]

lower_1 = [2000,0.001,1]
upper_1 = [10000,1,10]
lower_2 = [10000,0.001,7]
upper_2 = [15000,1,20]
lower_3 = [100000,0.001,20]
upper_3 = [300000,1,40]
p0_1 = [6000,0.2,3]
p0_2 = [12000,0.2,7]
p0_3 = [150000,0.2,40]

popt, pcov = curve_fit(sigmoid,x[:-2],data[:-2],p0=p0_1,bounds=(lower_1, upper_1),method='trf')
max_infected1 = popt[0]
error1 = sqrt(diag(pcov))[0]
fitted1 = [sigmoid(i,*popt) for i in t]
print(str(popt)+" +- "+str(error1))

popt, pcov = curve_fit(sigmoid,x[:-1],data[:-1],p0=p0_2,bounds=(lower_2, upper_2),method='trf')
max_infected2 = popt[0]
error2 = sqrt(diag(pcov))[0]
fitted2 = [sigmoid(i,*popt) for i in t]
print(str(popt)+" +- "+str(error2))

popt, pcov = curve_fit(sigmoid,x,data,p0=p0_3,bounds=(lower_3, upper_3),method='trf')
max_infected3 = popt[0]
error3 = sqrt(diag(pcov))[0]
fitted3 = [sigmoid(i,*popt) for i in t]
print(str(popt)+" +- "+str(error3))

popt, pcov = curve_fit(exponential,x,data,p0=[400,0.2],bounds=([100,0], [1000,2]),method='trf')
print(popt)
exp_fit = [exponential(i,*popt) for i in t]

plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="Epidemia inarrestabile")
plt.plot(t,fitted1,linestyle="--",zorder=2,label=bef_yesterday)
plt.plot(t,fitted2,linestyle="--",zorder=3,label=yesterday)
plt.plot(t,fitted3,zorder=4,label=date)
plt.scatter(x,data,marker="^",color="black",s=50,zorder=5)
plt.xlim(0,x[-1]+delta_t2)
plt.ylim(0,15000)
plt.title(date)
plt.xlabel("Tempo")
plt.ylabel("Persone Infette")
plt.legend()
plt.savefig("plot1.png",dpi=300,bbox_inches='tight')
plt.show()
plt.clf()

plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="$N_{MAX}=\infty$")
plt.plot(t,fitted1,linestyle="--",zorder=1,label="$N_{MAX}="+str(int(max_infected1))+"\pm"+str(round(error1/max_infected1,2))+"\%$")
plt.plot(t,fitted2,linestyle="--",zorder=2,label="$N_{MAX}="+str(int(max_infected2))+"\pm"+str(round(error2/max_infected2,2))+"\%$")
plt.plot(t,fitted3,zorder=3,label="$N_{MAX}="+str(int(max_infected3))+"\pm"+str(round(error3/max_infected3,2))+"\%$")
plt.scatter(x,data,marker="^",color="black",s=40,zorder=4)
plt.xlim(0,x[-1]+delta_t)
plt.ylim(0,max(fitted3))
plt.title(date)
plt.xlabel("Tempo")
plt.ylabel("Persone Infette")
plt.legend()
plt.savefig("plot2.png",dpi=300,bbox_inches='tight')
plt.show()
plt.clf()
