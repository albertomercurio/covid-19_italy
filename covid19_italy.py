import os
import csv
import matplotlib.pyplot as plt
from numpy import exp, linspace, sqrt, diag, diff
from scipy.optimize import curve_fit
import datetime

date = datetime.date.today()
yesterday = datetime.date.today() - datetime.timedelta(days=1)
bef_yesterday = datetime.date.today() - datetime.timedelta(days=2)

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
os.system("wget -O data.csv "+url)

infetti = []

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
            infetti.append(int(row[6]))

delta_t = 30
delta_t2 = 10
x = range(len(infetti))
x2 = range(len(infetti)+delta_t)
t = linspace(0,len(infetti)+delta_t,100)

lower_1 = [1000,0.001,0]
# upper_1 = [10000,1,10]
# lower_2 = [10000,0.001,7]
# upper_2 = [15000,1,20]
# lower_3 = [100000,0.001,20]
upper_3 = [500000,1,100]
# p0_1 = [6000,0.2,3]
# p0_2 = [12000,0.2,7]
# p0_3 = [150000,0.2,40]
p0 = [20000,(lower_1[1]+upper_3[1])/2,(lower_1[2]+upper_3[2])/2]
print(p0)

# popt, pcov = curve_fit(sigmoid,x[:-2],infetti[:-2],p0=p0,bounds=(lower_1, upper_3),method='trf',
# max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
# max_infected1 = popt[0]
# error1 = sqrt(diag(pcov))[0]
# fitted1 = [sigmoid(i,*popt) for i in t]
# print(str(popt)+" +- "+str(error1))
#
# popt, pcov = curve_fit(sigmoid,x[:-1],infetti[:-1],p0=p0,bounds=(lower_1, upper_3),method='trf',
# max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
# max_infected2 = popt[0]
# error2 = sqrt(diag(pcov))[0]
# fitted2 = [sigmoid(i,*popt) for i in t]
# print(str(popt)+" +- "+str(error2))

popt, pcov = curve_fit(sigmoid,x,infetti,p0=p0,bounds=(lower_1,upper_3),method='trf',
max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="huber")
max_infected3 = popt[0]
error3 = sqrt(diag(pcov))[0]
fitted3 = [sigmoid(i,*popt) for i in t]
print(str(popt)+" +- "+str(error3))

popt, pcov = curve_fit(exponential,x,infetti,p0=[400,0.2],bounds=([100,0], [1000,2]),method='trf',
max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="huber")
print(popt)
exp_fit = [exponential(i,*popt) for i in t]

plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="Epidemia inarrestabile")
# plt.plot(t,fitted1,linestyle="--",lw=0.8,zorder=2,label=bef_yesterday)
# plt.plot(t,fitted2,linestyle="--",lw=1,zorder=3,label=yesterday)
plt.plot(t,fitted3,color="red",zorder=4,label=date)
plt.scatter(x,infetti,marker="^",color="black",s=50,zorder=5)
plt.xlim(0,x[-1]+delta_t2)
plt.ylim(0,2.5*max(infetti))
plt.title(date)
plt.xlabel("Tempo (Giorni dal 24/02/2020)")
plt.ylabel("Persone infette")
plt.legend()
plt.savefig("img/"+str(date)+"_1.png",dpi=300,bbox_inches='tight')
plt.clf()

plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="$N_{MAX}=\infty$")
# plt.plot(t,fitted1,linestyle="--",lw=0.8,zorder=1,label="$N_{MAX}="+str(int(max_infected1))+"\pm"+str(round(error1/max_infected1*100,1))+"\%$")
# plt.plot(t,fitted2,linestyle="--",lw=1,zorder=2,label="$N_{MAX}="+str(int(max_infected2))+"\pm"+str(round(error2/max_infected2*100,1))+"\%$")
plt.plot(t,fitted3,zorder=3,color="red",label="$N_{MAX}="+str(int(max_infected3))+"\pm"+str(round(error3/max_infected3*100,1))+"\%$")
plt.scatter(x,infetti,marker="^",color="black",s=40,zorder=4)
plt.xlim(0,x[-1]+delta_t)
plt.ylim(0,max(fitted3))
plt.title(date)
plt.xlabel("Tempo (Giorni dal 24/02/2020)")
plt.ylabel("Persone infette")
plt.legend()
plt.savefig("img/"+str(date)+"_2.png",dpi=300,bbox_inches='tight')
plt.clf()

##################################################################
##################################################################
##################################################################

variazione_infetti = diff(infetti)

plt.plot(x[:-1],variazione_infetti,lw=1,color="blue",linestyle="-.")
plt.scatter(x[:-1],variazione_infetti,marker="^",color="black",s=40,zorder=4)
plt.xlabel("Tempo (Giorni dal 25/02/2020)")
plt.ylabel("Nuovi infetti")
plt.savefig("img/nuovi_infetti.png",dpi=300,bbox_inches='tight')
plt.clf()

growth_factor = [variazione_infetti[i+1]/variazione_infetti[i] for i in range(len(variazione_infetti)-1)]
costante = [1 for i in range(len(growth_factor))]

plt.plot(x[:-2],costante,color="gray",linestyle="--")
plt.plot(x[:-2],growth_factor,lw=1,color="blue",linestyle="-.")
plt.scatter(x[:-2],growth_factor,marker="^",color="black",s=40,zorder=4)
plt.xlabel("Tempo (Giorni dal 26/02/2020)")
plt.ylabel("Fattore di crescita")
plt.savefig("img/growth_factor.png",dpi=300,bbox_inches='tight')
plt.clf()
