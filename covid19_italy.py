import os
import csv
import matplotlib.pyplot as plt
from numpy import exp, linspace, sqrt, diag, diff
from scipy.optimize import curve_fit
import datetime

fit_regioni = True #Mettere False se non si vuole eseguire il fit delle singole regioni

date = datetime.date.today()
yesterday = datetime.date.today() - datetime.timedelta(days=1)
bef_yesterday = datetime.date.today() - datetime.timedelta(days=2)

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
os.system("wget -O data.csv "+url)

infetti = []
perc_morti = []

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
        elif len(row) == 12:
            infetti.append(int(row[-2]))
            perc_morti.append(100*float(row[-3])/float(row[-2]))

delta_t = 30
delta_t2 = 10
x = range(len(infetti))
x2 = range(len(infetti)+delta_t)
t = linspace(0,len(infetti)+delta_t,100)

lower_1 = [1000,0.01,0]
# upper_1 = [10000,1,10]
# lower_2 = [10000,0.001,7]
# upper_2 = [15000,1,20]
# lower_3 = [100000,0.001,20]
upper_3 = [100000,1,100]
# p0_1 = [6000,0.2,3]
# p0_2 = [12000,0.2,7]
# p0_3 = [150000,0.2,40]
p0 = [20000,(lower_1[1]+upper_3[1])/2,(lower_1[2]+upper_3[2])/2]
print(p0)

# popt, pcov = curve_fit(sigmoid,x[:-2],infetti[:-2],p0=p0,bounds=(lower_1, upper_3),method='trf',
# max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="huber")
# max_infected1 = popt[0]
# error1 = sqrt(diag(pcov))[0]
# fitted1 = [sigmoid(i,*popt) for i in t]
# print(str(popt)+" +- "+str(error1))
#
# popt, pcov = curve_fit(sigmoid,x[:-1],infetti[:-1],p0=p0,bounds=(lower_1, upper_3),method='trf',
# max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="huber")
# max_infected2 = popt[0]
# error2 = sqrt(diag(pcov))[0]
# fitted2 = [sigmoid(i,*popt) for i in t]
# print(str(popt)+" +- "+str(error2))

popt, pcov = curve_fit(sigmoid,x,infetti,p0=p0,bounds=(lower_1,upper_3),method='trf',
max_nfev=100000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="huber")
max_infected3 = popt[0]
error3 = sqrt(diag(pcov))[0]
fitted3 = [sigmoid(i,*popt) for i in t]
print(str(popt[0])+" +- "+str(error3))

popt2, pcov2 = curve_fit(exponential,x,infetti,p0=[400,0.2],bounds=([100,0], [10000,2]),method='trf',
max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="huber")
print(popt2)
exp_fit = [exponential(i,*popt2) for i in t]

plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="Epidemia inarrestabile")
# plt.plot(t,fitted1,linestyle="--",lw=0.8,zorder=2,label=bef_yesterday)
# plt.plot(t,fitted2,linestyle="--",lw=1,zorder=3,label=yesterday)
plt.plot(t,fitted3,color="red",zorder=4,label=date)
plt.scatter(x,infetti,marker="^",color="black",s=50,zorder=5)
plt.xlim(0,x[-1]+delta_t2)
plt.ylim(0,2.5*max(infetti))
plt.title(str(date)+" in Italia")
plt.xlabel("Tempo (Giorni dal 24/02/2020)")
plt.ylabel("Persone infette")
plt.legend()
plt.savefig("img_italia/"+str(date)+"_1.png",dpi=200,bbox_inches='tight')
plt.clf()

plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="$N_{MAX}=\infty$")
# plt.plot(t,fitted1,linestyle="--",lw=0.8,zorder=1,label="$N_{MAX}="+str(int(max_infected1))+"\pm"+str(round(error1/max_infected1*100,1))+"\%$")
# plt.plot(t,fitted2,linestyle="--",lw=1,zorder=2,label="$N_{MAX}="+str(int(max_infected2))+"\pm"+str(round(error2/max_infected2*100,1))+"\%$")
plt.plot(t,fitted3,zorder=3,color="red",label="$N_{MAX}="+str(int(max_infected3))+"\pm"+str(round(error3/max_infected3*100,1))+"\%$")
plt.scatter(x,infetti,marker="^",color="black",s=40,zorder=4)
plt.xlim(0,x[-1]+delta_t)
plt.ylim(0,max(fitted3))
plt.title(str(date)+" in Italia")
plt.xlabel("Tempo (Giorni dal 24/02/2020)")
plt.ylabel("Persone infette")
plt.legend()
plt.savefig("img_italia/"+str(date)+"_2.png",dpi=200,bbox_inches='tight')
plt.clf()

##################################################################

variazione_infetti = diff(infetti)
variazione_teorica = diff(fitted3)/(t[1]-t[0])

plt.plot(x[:-1],variazione_infetti,lw=1,color="blue",linestyle="-.",zorder=1)
plt.plot(t[:-1],variazione_teorica,color="red",zorder=2)
plt.scatter(x[:-1],variazione_infetti,marker="^",color="black",s=40,zorder=4)
plt.xlabel("Tempo (Giorni dal 25/02/2020)")
plt.ylabel("Nuovi infetti")
plt.xlim(0,t[-1])
plt.title(str(date)+" in Italia")
plt.savefig("img_italia/nuovi_infetti.png",dpi=200,bbox_inches='tight')
plt.clf()


# growth_factor = [variazione_infetti[i+1]/variazione_infetti[i] for i in range(len(variazione_infetti)-1)]
growth_factor = [float(infetti[i+1])/(infetti[i]) for i in range(len(infetti)-1)]
costante = [1 for i in range(len(growth_factor))]

plt.plot(x[:-1],costante,color="gray",linestyle="--")
plt.plot(x[:-1],growth_factor,lw=1,color="blue",linestyle="-.")
plt.scatter(x[:-1],growth_factor,marker="^",color="black",s=40,zorder=4)
plt.xlabel("Tempo (Giorni dal 26/02/2020)")
plt.ylabel("Fattore di crescita")
plt.title(str(date)+" in Italia")
plt.savefig("img_italia/growth_factor.png",dpi=200,bbox_inches='tight')
plt.clf()

plt.plot(x,perc_morti,lw=1,color="blue",linestyle="-.")
plt.scatter(x,perc_morti,marker="^",color="black",s=40,zorder=4)
plt.xlabel("Tempo (Giorni dal 26/02/2020)")
plt.ylabel("Tasso di mortalità (%)")
plt.title(str(date)+" in Italia")
plt.savefig("img_italia/death_rate.png",dpi=200,bbox_inches='tight')
plt.clf()


##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

max_infected = max_infected3
err_infected = error3

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
os.system("wget -O data.csv "+url)

regioni = ["Abruzzo","Basilicata","P.A. Bolzano","Calabria","Campania","Emilia Romagna","Friuli Venezia Giulia",
"Lazio","Liguria","Lombardia","Marche","Molise","Piemonte","Puglia","Sardegna","Sicilia","Toscana","P.A. Trento",
"Umbria","Valle d'Aosta","Veneto"]
hist = []

if fit_regioni:
    for regione in regioni:

        infetti = []
        directory = ("img_"+regione).replace(" ","").lower()

        if not os.path.exists(directory):
            os.mkdir(directory)

        with open('data.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first_row = True
            for row in csv_reader:
                if first_row:
                    first_row = False
                elif row[3] == regione:
                    infetti.append(int(row[-2]))


        print(regione)

        delta_t = 30
        delta_t2 = 10
        x = range(len(infetti))
        x2 = range(len(infetti)+delta_t)
        t = linspace(0,len(infetti)+delta_t,100)

        lower_1 = [10,0.01,1]
        upper_3 = [max_infected+3*err_infected,1.2,150]
        p0 = [2000,0.5,20]

        popt, pcov = curve_fit(sigmoid,x,infetti,p0=p0,bounds=(lower_1,upper_3),method='trf',
        max_nfev=100000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
        max_infected3 = popt[0]
        error3 = sqrt(diag(pcov))[0]
        fitted3 = [sigmoid(i,*popt) for i in t]
        print(str(popt)+" +- "+str(error3))
        print("############")

        popt, pcov = curve_fit(exponential,x,infetti,p0=[400,0.2],bounds=([0.001,0], [10000,2]),method='trf',
        max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="huber")
        exp_fit = [exponential(i,*popt) for i in t]

        plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="Epidemia inarrestabile")
        plt.plot(t,fitted3,color="red",zorder=4,label=date)
        plt.scatter(x,infetti,marker="^",color="black",s=50,zorder=5)
        plt.xlim(0,x[-1]+delta_t2)
        plt.ylim(0,2.5*max(infetti))
        plt.title(str(date)+" in "+regione)
        plt.xlabel("Tempo (Giorni dal 24/02/2020)")
        plt.ylabel("Persone infette")
        plt.legend()
        plt.savefig(directory+"/"+str(date)+"_1.png",dpi=200,bbox_inches='tight')
        plt.clf()

        plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="$N_{MAX}=\infty$")
        plt.plot(t,fitted3,zorder=3,color="red",label="$N_{MAX}="+str(int(max_infected3))+"\pm"+str(round(error3/max_infected3*100,1))+"\%$")
        plt.scatter(x,infetti,marker="^",color="black",s=40,zorder=4)
        plt.xlim(0,x[-1]+delta_t)
        plt.ylim(0,max(fitted3))
        plt.title(str(date)+" in "+regione)
        plt.xlabel("Tempo (Giorni dal 24/02/2020)")
        plt.ylabel("Persone infette")
        plt.legend()
        plt.savefig(directory+"/"+str(date)+"_2.png",dpi=200,bbox_inches='tight')
        plt.clf()

        ##################################################################

        variazione_infetti = diff(infetti)
        variazione_teorica = diff(fitted3)/(t[1]-t[0])

        plt.plot(x[:-1],variazione_infetti,lw=1,color="blue",linestyle="-.",zorder=1)
        plt.plot(t[:-1],variazione_teorica,color="red",zorder=2)
        plt.scatter(x[:-1],variazione_infetti,marker="^",color="black",s=40,zorder=4)
        plt.xlabel("Tempo (Giorni dal 25/02/2020)")
        plt.ylabel("Nuovi infetti")
        plt.title(str(date)+" in "+regione)
        plt.savefig(directory+"/"+"nuovi_infetti.png",dpi=200,bbox_inches='tight')
        plt.clf()

        for item in range(len(infetti)):
            infetti[item] = max(infetti[item],1)

        growth_factor = [float(infetti[i+1])/(infetti[i]) for i in range(len(infetti)-1)]
        costante = [1 for i in range(len(growth_factor))]

        plt.plot(x[:-1],costante,color="gray",linestyle="--")
        plt.plot(x[:-1],growth_factor,lw=1,color="blue",linestyle="-.")
        plt.scatter(x[:-1],growth_factor,marker="^",color="black",s=40,zorder=4)
        plt.xlabel("Tempo (Giorni dal 26/02/2020)")
        plt.ylabel("Fattore di crescita")
        plt.title(str(date)+" in "+regione)
        plt.savefig(directory+"/""growth_factor.png",dpi=200,bbox_inches='tight')
        plt.clf()

        hist.append(infetti[-1])

for i in range(len(regioni)):
    plt.bar(i,hist[i])
plt.ylabel("Totale infetti")
plt.xticks(range(len(regioni)),regioni,rotation="vertical")
plt.title(str(date)+" in Italia")
plt.savefig("img_italia/"+str(date)+"_regioni.png",dpi=200,bbox_inches='tight')
plt.clf()
