import os
import csv
import matplotlib.pyplot as plt
from numpy import exp, linspace, arange, sqrt, diag, diff, mean, sum, ndarray, add, subtract
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import datetime
import geopandas as gpd
import pandas as pd

fit_regioni = True #Mettere False se non si vuole eseguire il fit delle singole regioni
plot_gompertz = False
plot_richards = True
plot_sigmoid = False
plot_sir = True
plot_map = True

N = 60483973 #Popolazione italiana

date = datetime.date.today()
yesterday = datetime.date.today() - datetime.timedelta(days=1)
bef_yesterday = datetime.date.today() - datetime.timedelta(days=2)

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
os.system("wget -O data.csv "+url)

infetti = []
perc_morti = []
morti = []

#Lockdown
def lkd(t,a1,a2,a3=15):
    if t < a3:
        return 1
    else:
        return (1 - a2)*exp(-(t-a3)/a1) + a2

def deriv(u, t, a1, a2, b, g):
    x, y, z = u
    return [-lkd(t,a1,a2)*b*x*y, lkd(t,a1,a2)*b*x*y - g*y, g*y]

def sigmoid(x, a, b, c):
    return a/(1 + exp(-b*(x-c)))

def exponential(x, a, b):
    return a*exp(b*x)

def gompertz(x, a, b, c):
    return a*exp(-exp(b-c*x))

def richards(x, a, b, c, d=0.2):
    global e
    return a/((1 + ((a/e)**d - 1)*exp(-b*(x-c)))**(1.0/d))

def sir(x, b, g, a1, a2):
    global y0
    dt = 0.001
    if not isinstance(x, (list, tuple, ndarray,range)):
        if x == 0:
            x = 1e-20
        t = arange(0, abs(x), dt)
        if x < 0:
            t = t[::-1]
        sol = odeint(deriv, y0, t, args=(a1,a2,b,g))
        # sol = odeint(deriv, y0, t, args=(b,g))
        s = [i[0] for i in sol]
        i = [i[1] for i in sol]
        r = [i[2] for i in sol]
        return i[-1]+r[-1]
    else:
        result = []
        for it in x:
            if it == 0:
                it = 1e-20
            t = arange(0, abs(it), dt)
            if it < 0:
                t = t[::-1]
            sol = odeint(deriv, y0, t, args=(a1,a2,b,g))
            # sol = odeint(deriv, y0, t, args=(b,g))
            s = [i[0] for i in sol]
            i = [i[1] for i in sol]
            r = [i[2] for i in sol]
            result.append(i[-1]+r[-1])
        return result

def r_sqrt(data,data_fitted):
    if len(data) == len(data_fitted):
        ESS = sum([(data_fitted[i]-mean(data))**2 for i in range(len(data_fitted))])
        RSS = sum([(data_fitted[i]-data[i])**2 for i in range(len(data_fitted))])
        TSS = sum([(data[i]-mean(data))**2 for i in range(len(data))])
        return round(1-RSS/TSS,5)
        # return round(ESS/TSS,5)
    else:
        print("Error: data and data_fitted doesn't have the same dimension.")
        return 0

with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first_row = True
    for row in csv_reader:
        if first_row:
            first_row = False
        elif len(row) == 15:
            infetti.append(int(row[-4]))
            perc_morti.append(100*float(row[-5])/float(row[-4]))
            morti.append(int(row[-5]))

delta_t = 30
delta_t2 = 10
x = range(len(infetti))
x2 = range(len(infetti)+delta_t)
t = linspace(0,len(infetti)+delta_t,100)

###################################
# sigmoid
###################################
if plot_sigmoid:
    print("###################################")
    print("Sigmoid")
    print("###################################")

    print("----INFETTI----")
    lower = [60000,0.01,0]
    upper = [500000,1,100]
    p0 = [130000,0.2,30]
    print(p0)

    popt, pcov = curve_fit(sigmoid,x[0:],infetti[0:],p0=p0,bounds=(lower,upper),method='trf',
    max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
    max_infected_sig = popt[0]
    error_sigmoid = sqrt(diag(pcov))[0]
    fitted_sig = [sigmoid(i,*popt) for i in t]
    r2_sig = r_sqrt(infetti,[sigmoid(i,*popt) for i in x])
    print(str(popt)+" +- "+str(int(error_sigmoid)))
    print("\n R^2 ="+str(r2_sig)+"\n")

    print("----DECEDUTI----")
    lower = [4000,0.01,0]
    upper = [80000,1,100]
    p0 = [10000,0.2,30]
    print(p0)

    popt, pcov = curve_fit(sigmoid,x[0:],morti[0:],p0=p0,bounds=(lower,upper),method='trf',
    max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
    max_dead_sig = popt[0]
    error_sigmoid_dead = sqrt(diag(pcov))[0]
    fitted_sig_dead = [sigmoid(i,*popt) for i in t]
    r2_sig_dead = r_sqrt(morti,[sigmoid(i,*popt) for i in x])
    print(str(popt)+" +- "+str(int(error_sigmoid_dead)))
    print("\n R^2 ="+str(r2_sig_dead)+"\n")

###################################
# gompertz
###################################
if plot_gompertz:
    print("###################################")
    print("Gompertz")
    print("###################################")

    print("----INFETTI----")
    lower = [80000,1.1,0.01]
    upper = [1000000,20,2.0]
    p0 = [500000,2.68,0.1]
    print(p0)

    popt, pcov = curve_fit(gompertz,x[0:],infetti[0:],p0=p0,bounds=(lower,upper),method='trf',
    max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
    max_infected_gom = popt[0]
    error_gompertz = sqrt(diag(pcov))[0]
    fitted_gom = [gompertz(i,*popt) for i in t]
    r2_gom = r_sqrt(infetti,[gompertz(i,*popt) for i in x])
    print(str(popt)+" +- "+str(int(error_gompertz)))
    print("\n R^2 ="+str(r2_gom)+"\n")

    print("----DECEDUTI----")
    lower = [4000,1.1,0.01]
    upper = [100000,20,2.0]
    p0 = [10000,5,0.1]
    print(p0)

    popt, pcov = curve_fit(gompertz,x[0:],morti[0:],p0=p0,bounds=(lower,upper),method='trf',
    max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
    max_dead_gom = popt[0]
    error_gompertz_dead = sqrt(diag(pcov))[0]
    fitted_gom_dead = [gompertz(i,*popt) for i in t]
    r2_gom_dead = r_sqrt(morti,[gompertz(i,*popt) for i in x])
    print(str(popt)+" +- "+str(int(error_gompertz_dead)))
    print("\n R^2 ="+str(r2_gom_dead)+"\n")

###################################
# richards
###################################
if plot_richards:
    print("###################################")
    print("Richards")
    print("###################################")

    print("----INFETTI---- ")

    e = infetti[0]

    lower = [50000,0.01,-50,1e-20]
    upper = [1000000,1,0,5.0]
    p0 = [130000,0.2,-1,1.5]
    print(p0)

    popt, pcov = curve_fit(richards,x[0:],infetti[0:],p0=p0,bounds=(lower,upper),method='trf',
    max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
    max_infected_richards = popt[0]
    error_richards = sqrt(diag(pcov))[0]
    fitted_richards = [richards(i,*popt) for i in t]
    r2_richards = r_sqrt(infetti,[richards(i,*popt) for i in x])
    print(str(popt)+" +- "+str(int(error_richards)))
    print("\n R^2 ="+str(r2_richards)+"\n")

    print("----DECEDUTI---- ")

    e = morti[0]

    lower = [5000,0.01,-50,1e-20]
    upper = [100000,1,0,5.0]
    p0 = [13000,0.1,-1,1.5]
    print(p0)

    popt, pcov = curve_fit(richards,x[0:],morti[0:],p0=p0,bounds=(lower,upper),method='trf',
    max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
    max_dead_richards = popt[0]
    error_richards_dead = sqrt(diag(pcov))[0]
    fitted_richards_dead = [richards(i,*popt) for i in t]
    r2_richards_dead = r_sqrt(morti,[richards(i,*popt) for i in x])
    print(str(popt)+" +- "+str(int(error_richards_dead)))
    print("\n R^2 ="+str(r2_richards_dead)+"\n")

###################################
# sir
###################################
if plot_sir:
    print("###################################")
    print("SIR")
    print("###################################")

    print("----INFETTI---- ")

    N = 60483973 #Popolazione italiana
    y0 = [N-infetti[0],infetti[0],0]

    lower = [0,0,1,0.001]
    upper = [10,1,100,0.9]
    p0 = [0.5/N,0.18,20.0,0.3]
    print(p0)

    popt, pcov = curve_fit(sir,x[0:],infetti[0:],p0=p0,bounds=(lower,upper),method='trf',
    max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
    fitted_sir = [sir(i,*popt) for i in t]
    max_infected_sir = sir(200,*popt)
    error_sir = max_infected_sir*mean([sqrt(diag(pcov))[i]/popt[i] for i in range(len(popt))])
    # error_sir = max_infected_sir - (sir(200,*add(popt,sqrt(diag(pcov)))) + sir(200,*subtract(popt,sqrt(diag(pcov)))))/2
    r2_sir = r_sqrt(infetti,[sir(i,*popt) for i in x])
    print(str(popt)+" +- "+str(sqrt(diag(pcov))))
    print("\n R^2 ="+str(r2_sir)+"\n")

###################################
# exponential
###################################
print("###################################")
print("Exponential")
print("###################################")

print("----INFETTI----")
popt2, pcov2 = curve_fit(exponential,x[:25],infetti[:25],p0=[400,0.2],bounds=([100,0], [10000,2]),method='trf',
max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
print(popt2)
exp_fit = [exponential(i,*popt2) for i in t]
r2_exp = r_sqrt(infetti,[exponential(i,*popt2) for i in x])
print("\n R^2 ="+str(r2_exp)+"\n")

print("----DECEDUTI----")
popt2, pcov2 = curve_fit(exponential,x,morti,p0=[400,0.2],bounds=([10,0], [1000,2]),method='trf',
max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
print(popt2)
exp_fit_dead = [exponential(i,*popt2) for i in t]
r2_exp_dead = r_sqrt(morti,[exponential(i,*popt2) for i in x])
print("\n R^2 ="+str(r2_exp_dead)+"\n")

##################################################################

plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="Epidemia inarrestabile ($R^2="+str(r2_exp)+")$")
if plot_sir: plt.plot(t,fitted_sir,color="purple",zorder=4,label="S.I.R. Mod. ($R^2="+str(r2_sir)+")$")
if plot_richards: plt.plot(t,fitted_richards,color="#FFCC00",zorder=3,label="Richards ($R^2="+str(r2_richards)+")$")
if plot_gompertz: plt.plot(t,fitted_gom,color="#228B22",zorder=3,label="Gompertz ($R^2="+str(r2_gom)+")$")
if plot_sigmoid: plt.plot(t,fitted_sig,color="red",zorder=3,label="Sigmoide ($R^2="+str(r2_sig)+")$")
plt.scatter(x,infetti,marker="^",color="black",s=50,zorder=5)
plt.xlim(0,x[-1]+delta_t2)
plt.ylim(0,2.5*max(infetti))
plt.title(str(date)+" in Italia")
plt.xlabel("Tempo (Giorni dal 24/02/2020)")
plt.ylabel("Persone infette")
plt.legend()
plt.savefig("img_italia/"+str(date)+"_1.png",dpi=200,bbox_inches='tight')
plt.clf()

plt.plot(t,exp_fit_dead,linestyle="-.",zorder=1,label="Epidemia inarrestabile ($R^2="+str(r2_exp_dead)+")$")
if plot_richards: plt.plot(t,fitted_richards_dead,color="#FFCC00",zorder=3,label="Richards ($R^2="+str(r2_richards_dead)+")$")
if plot_gompertz: plt.plot(t,fitted_gom_dead,color="#228B22",zorder=3,label="Gompertz ($R^2="+str(r2_gom_dead)+")$")
if plot_sigmoid: plt.plot(t,fitted_sig_dead,color="red",zorder=3,label="Sigmoide ($R^2="+str(r2_sig_dead)+")$")
plt.scatter(x,morti,marker="^",color="black",s=50,zorder=5)
plt.xlim(0,x[-1]+delta_t2)
plt.ylim(0,2.5*max(morti))
plt.title(str(date)+" in Italia")
plt.xlabel("Tempo (Giorni dal 24/02/2020)")
plt.ylabel("Persone decedute")
plt.legend()
plt.savefig("img_italia/"+str(date)+"_dead_1.png",dpi=200,bbox_inches='tight')
plt.clf()

##################################################################

plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="$N_{MAX}=\infty$")
if plot_sir: plt.plot(t,fitted_sir,zorder=3,color="purple",label="$N_{MAX}="+str(int(max_infected_sir))+
"\pm"+str(round(error_sir/max_infected_sir*100,1))+"\%$")
if plot_richards: plt.plot(t,fitted_richards,zorder=3,color="#FFCC00",label="$N_{MAX}="+str(int(max_infected_richards))+
"\pm"+str(round(error_richards/max_infected_richards*100,1))+"\%$")
if plot_gompertz: plt.plot(t,fitted_gom,zorder=3,color="#228B22",label="$N_{MAX}="+str(int(max_infected_gom))+
"\pm"+str(round(error_gompertz/max_infected_gom*100,1))+"\%$")
if plot_sigmoid: plt.plot(t,fitted_sig,zorder=3,color="red",label="$N_{MAX}="+str(int(max_infected_sig))+
"\pm"+str(round(error_sigmoid/max_infected_sig*100,1))+"\%$")
plt.scatter(x,infetti,marker="^",color="black",s=40,zorder=4)
plt.xlim(0,x[-1]+delta_t)
# plt.ylim(0,max(max(fitted_sig),max(fitted_gom)))
plt.ylim(0,2.5*max(infetti))
plt.title(str(date)+" in Italia")
plt.xlabel("Tempo (Giorni dal 24/02/2020)")
plt.ylabel("Persone infette")
plt.legend()
plt.savefig("img_italia/"+str(date)+"_2.png",dpi=200,bbox_inches='tight')
plt.clf()

plt.plot(t,exp_fit_dead,linestyle="-.",zorder=1,label="$N_{MAX}=\infty$")
if plot_richards: plt.plot(t,fitted_richards_dead,zorder=3,color="#FFCC00",label="$N_{MAX}="+str(int(max_dead_richards))+
"\pm"+str(round(error_richards_dead/max_dead_richards*100,1))+"\%$")
if plot_gompertz: plt.plot(t,fitted_gom_dead,zorder=3,color="#228B22",label="$N_{MAX}="+str(int(max_dead_gom))+
"\pm"+str(round(error_gompertz_dead/max_dead_gom*100,1))+"\%$")
if plot_sigmoid: plt.plot(t,fitted_sig_dead,zorder=3,color="red",label="$N_{MAX}="+str(int(max_dead_sig))+
"\pm"+str(round(error_sigmoid_dead/max_dead_sig*100,1))+"\%$")
plt.scatter(x,morti,marker="^",color="black",s=40,zorder=4)
plt.xlim(0,x[-1]+delta_t)
# plt.ylim(0,max(max(fitted_sig_dead),max(fitted_gom_dead)))
plt.ylim(0,2.5*max(morti))
plt.title(str(date)+" in Italia")
plt.xlabel("Tempo (Giorni dal 24/02/2020)")
plt.ylabel("Persone decedute")
plt.legend()
plt.savefig("img_italia/"+str(date)+"_dead_2.png",dpi=200,bbox_inches='tight')
plt.clf()

##################################################################

variazione_infetti = diff(infetti)
if plot_sigmoid: variazione_teorica_sig = diff(fitted_sig)/(t[1]-t[0])
if plot_gompertz: variazione_teorica_gom = diff(fitted_gom)/(t[1]-t[0])
if plot_richards: variazione_teorica_richards = diff(fitted_richards)/(t[1]-t[0])
if plot_sir: variazione_teorica_sir = diff(fitted_sir)/(t[1]-t[0])

variazione_morti = diff(morti)
if plot_sigmoid: variazione_teorica_sig_dead = diff(fitted_sig_dead)/(t[1]-t[0])
if plot_gompertz: variazione_teorica_gom_dead = diff(fitted_gom_dead)/(t[1]-t[0])
if plot_richards: variazione_teorica_richards_dead = diff(fitted_richards_dead)/(t[1]-t[0])

##################################################################

plt.plot(x[:-1],variazione_infetti,lw=1,color="blue",linestyle="-.",zorder=1)
if plot_sir: plt.plot(t[:-1],variazione_teorica_sir,color="purple",zorder=3)
if plot_richards: plt.plot(t[:-1],variazione_teorica_richards,color="#FFCC00",zorder=2)
if plot_gompertz: plt.plot(t[:-1],variazione_teorica_gom,color="#228B22",zorder=2)
if plot_sigmoid: plt.plot(t[:-1],variazione_teorica_sig,color="red",zorder=2)
plt.scatter(x[:-1],variazione_infetti,marker="^",color="black",s=40,zorder=4)
plt.xlabel("Tempo (Giorni dal 25/02/2020)")
plt.ylabel("Nuovi infetti")
plt.xlim(0,t[-1])
plt.ylim(0,2*max(variazione_infetti))
plt.title(str(date)+" in Italia")
plt.savefig("img_italia/nuovi_infetti.png",dpi=200,bbox_inches='tight')
plt.clf()

plt.plot(x[:-1],variazione_morti,lw=1,color="blue",linestyle="-.",zorder=1)
if plot_richards: plt.plot(t[:-1],variazione_teorica_richards_dead,color="#FFCC00",zorder=2)
if plot_gompertz: plt.plot(t[:-1],variazione_teorica_gom_dead,color="#228B22",zorder=2)
if plot_sigmoid: plt.plot(t[:-1],variazione_teorica_sig_dead,color="red",zorder=2)
plt.scatter(x[:-1],variazione_morti,marker="^",color="black",s=40,zorder=4)
plt.xlabel("Tempo (Giorni dal 25/02/2020)")
plt.ylabel("Nuovi deceduti")
plt.xlim(0,t[-1])
plt.ylim(0,2*max(variazione_morti))
plt.title(str(date)+" in Italia")
plt.savefig("img_italia/nuovi_deceduti.png",dpi=200,bbox_inches='tight')
plt.clf()

##################################################################

growth_factor = [float(infetti[i+1])/(infetti[i]) for i in range(len(infetti)-1)]
costante = [1 for i in range(len(growth_factor))]

growth_factor_dead = [float(morti[i+1])/(morti[i]) for i in range(len(morti)-1)]

##################################################################

plt.plot(x[:-1],costante,color="gray",linestyle="--")
plt.plot(x[:-1],growth_factor,lw=1,color="blue",linestyle="-.")
plt.scatter(x[:-1],growth_factor,marker="^",color="black",s=40,zorder=4)
plt.xlabel("Tempo (Giorni dal 26/02/2020)")
plt.ylabel("Fattore di crescita")
plt.title(str(date)+" in Italia")
plt.savefig("img_italia/growth_factor.png",dpi=200,bbox_inches='tight')
plt.clf()

plt.plot(x[:-1],costante,color="gray",linestyle="--")
plt.plot(x[:-1],growth_factor_dead,lw=1,color="blue",linestyle="-.")
plt.scatter(x[:-1],growth_factor_dead,marker="^",color="black",s=40,zorder=4)
plt.xlabel("Tempo (Giorni dal 26/02/2020)")
plt.ylabel("Fattore di crescita (deceduti)")
plt.title(str(date)+" in Italia")
plt.savefig("img_italia/growth_factor_dead.png",dpi=200,bbox_inches='tight')
plt.clf()

##################################################################

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

if fit_regioni:
    max_infected = 3*max(infetti)

    max_dead = 3*max(morti)

    url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
    os.system("wget -O data_regioni.csv "+url)

    regioni = ["Sicilia","Abruzzo","Basilicata","P.A. Bolzano","Calabria","Campania","Emilia Romagna","Friuli Venezia Giulia",
    "Lazio","Liguria","Lombardia","Marche","Molise","Piemonte","Puglia","Sardegna","Toscana","P.A. Trento",
    "Umbria","Valle d'Aosta","Veneto"]
    abitanti_regioni = [5026989,1315196,567118,106951,1956687,5826860,4452629,1215538,5896693,1556981,10036258,
    1531753,308493,4375865,4048242,1648176,3736968,1070340,884640,126202,4905037]
    hist = []
    hist2 = []
    it_ab = 0

    for regione in regioni:

        infetti = []
        morti = []
        directory = ("img_"+regione).replace(" ","").lower()

        if not os.path.exists(directory):
            os.mkdir(directory)

        with open('data_regioni.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first_row = True
            for row in csv_reader:
                if first_row:
                    first_row = False
                elif row[3].replace("-"," ") == regione:
                    infetti.append(int(row[-4]))
                    morti.append(int(row[-5]))

        print("//////////////// "+regione+" ////////////////")

        inizio_infetti = 0
        inizio_morti = 0

        for i in range(len(infetti)):
            if infetti[i] > 0:
                inizio_infetti = i
                break

        for i in range(len(morti)):
            if morti[i] > 0:
                inizio_morti = i
                break

        delta_t = 30
        delta_t2 = 10
        x = range(len(infetti))
        x2 = range(len(infetti)+delta_t)
        t = linspace(0,len(infetti)+delta_t,100)

        ###################################
        # sigmoid
        ###################################
        if plot_sigmoid:
            print("----INFETTI----")
            lower = [10,0.01,1]
            upper = [max_infected,1.2,150]
            p0 = [2000,0.5,20]

            if inizio_infetti != 0:
                popt, pcov = curve_fit(sigmoid,x[:-inizio_infetti],infetti[inizio_infetti:],p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=100000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            else:
                popt, pcov = curve_fit(sigmoid,x,infetti,p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=100000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")

            max_infected_sig = popt[0]
            error_sigmoid = sqrt(diag(pcov))[0]
            fitted_sig = [sigmoid(i-inizio_infetti,*popt) for i in t]
            r2_sig = r_sqrt(infetti,[sigmoid(i-inizio_infetti,*popt) for i in x])
            print(str(popt)+" +- "+str(int(error_sigmoid)))
            print("\n R^2 ="+str(r2_sig)+"\n")

            print("----DECEDUTI----")
            lower = [0,0.01,1]
            upper = [max_dead,10.0,150]
            p0 = [50,0.5,20]

            if inizio_morti != 0:
                popt, pcov = curve_fit(sigmoid,x[:-inizio_morti],morti[inizio_morti:],p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            else:
                popt, pcov = curve_fit(sigmoid,x,morti,p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            max_dead_sig = popt[0]
            error_sigmoid_dead = sqrt(diag(pcov))[0]
            fitted_sig_dead = [sigmoid(i-inizio_morti,*popt) for i in t]
            r2_sig_dead = r_sqrt(morti,[sigmoid(i-inizio_morti,*popt) for i in x])
            print(str(popt)+" +- "+str(int(error_sigmoid_dead)))
            print("\n R^2 ="+str(r2_sig_dead)+"\n")

        ###################################
        # gompertz
        ###################################
        if plot_gompertz:
            print("----INFETTI----")
            lower = [10,0.005,0.005]
            upper = [max_infected,50,0.5]
            p0 = [1000,2.68,0.1]

            if inizio_infetti != 0:
                popt, pcov = curve_fit(gompertz,x[:-inizio_infetti],infetti[inizio_infetti:],p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=200000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            else:
                popt, pcov = curve_fit(gompertz,x,infetti,p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=200000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            max_infected_gom = popt[0]
            error_gompertz = sqrt(diag(pcov))[0]
            fitted_gom = [gompertz(i-inizio_infetti,*popt) for i in t]
            r2_gom = r_sqrt(infetti,[gompertz(i-inizio_infetti,*popt) for i in x])
            print(str(popt)+" +- "+str(int(error_gompertz)))
            print("\n R^2 ="+str(r2_gom)+"\n")

            print("----DECEDUTI----")
            lower = [0,0.005,0.005]
            upper = [max_dead,50,1.5]
            p0 = [100,5,0.1]

            if inizio_morti != 0:
                popt, pcov = curve_fit(gompertz,x[:-inizio_morti],morti[inizio_morti:],p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            else:
                popt, pcov = curve_fit(gompertz,x,morti,p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            max_dead_gom = popt[0]
            error_gompertz_dead = sqrt(diag(pcov))[0]
            fitted_gom_dead = [gompertz(i-inizio_morti,*popt) for i in t]
            r2_gom_dead = r_sqrt(morti,[gompertz(i-inizio_morti,*popt) for i in x])
            print(str(popt)+" +- "+str(int(error_gompertz_dead)))
            print("\n R^2 ="+str(r2_gom_dead)+"\n")

        ###################################
        # richards
        ###################################
        if plot_richards:

            print("----INFETTI----")

            e = infetti[inizio_infetti]

            lower = [10,0.01,-40,1e-30]
            upper = [max_infected,3.0,20,5.0]
            p0 = [1000,0.1,1,0.1]

            if inizio_infetti != 0:
                popt, pcov = curve_fit(richards,x[:-inizio_infetti],infetti[inizio_infetti:],p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            else:
                popt, pcov = curve_fit(richards,x,infetti,p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            max_infected_richards = popt[0]
            error_richards = sqrt(diag(pcov))[0]
            fitted_richards = [richards(i-inizio_infetti,*popt) for i in t]
            r2_richards = r_sqrt(infetti,[richards(i-inizio_infetti,*popt) for i in x])
            print(str(popt)+" +- "+str(error_richards))
            print("\n R^2 ="+str(r2_richards)+"\n")

            print("----DECEDUTI----")

            e = morti[inizio_morti]

            lower = [0,0.01,-40,1e-30]
            upper = [max_dead,3.0,100,5.0]
            p0 = [100,0.1,1.5,0.1]

            if inizio_morti != 0:
                popt, pcov = curve_fit(richards,x[:-inizio_morti],morti[inizio_morti:],p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            else:
                popt, pcov = curve_fit(richards,x,morti,p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            max_dead_richards = popt[0]
            error_richards_dead = sqrt(diag(pcov))[0]
            fitted_richards_dead = [richards(i-inizio_morti,*popt) for i in t]
            r2_richards_dead = r_sqrt(morti,[richards(i-inizio_morti,*popt) for i in x])
            try:
                print(str(popt)+" +- "+str(int(error_richards_dead)))
            except:
                print(str(popt)+" +- "+str(error_richards_dead))
            print("\n R^2 ="+str(r2_richards_dead)+"\n")

        ###################################
        # sir
        ###################################
        plot_sir = False
        if plot_sir:

            print("----INFETTI----")

            N = abitanti_regioni[it_ab]
            y0 = [N-infetti[inizio_infetti],infetti[inizio_infetti],0]

            lower = [0,0,10,1e-20]
            upper = [100,1,100,0.9]
            p0 = [50.0/N,0.15,40,0.1]

            if inizio_infetti != 0:
                popt, pcov = curve_fit(sir,x[:-inizio_infetti],infetti[inizio_infetti:],p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            else:
                popt, pcov = curve_fit(sir,x,infetti,p0=p0,bounds=(lower,upper),method='trf',
                max_nfev=1000000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
            fitted_sir = [sir(i-inizio_infetti,*popt) for i in t]
            max_infected_sir = sir(200,*popt)
            error_sir = max_infected_sir*mean([sqrt(diag(pcov))[i]/popt[i] for i in range(len(popt))])
            # error_sir = max_infected_sir - (sir(200,*add(popt,sqrt(diag(pcov)))) + sir(200,*subtract(popt,sqrt(diag(pcov)))))/2
            r2_sir = r_sqrt(infetti,[sir(i-inizio_infetti,*popt) for i in x])
            print(str(popt)+" +- "+str(sqrt(diag(pcov))))
            print("\n R^2 ="+str(r2_sir)+"\n")

        ###################################
        # exponential
        ###################################

        popt, pcov = curve_fit(exponential,x[:20],infetti[:20],p0=[400,0.2],bounds=([0.001,0], [10000,2]),method='trf',
        max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
        exp_fit = [exponential(i-inizio_infetti,*popt) for i in t]
        r2_exp = r_sqrt(infetti,[exponential(i-inizio_infetti,*popt) for i in x])

        if inizio_morti != 0:
            popt, pcov = curve_fit(exponential,x[:-inizio_morti],morti[inizio_morti:],p0=[400,0.2],bounds=([0.001,0], [10000,2]),method='trf',
            max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
        else:
            popt, pcov = curve_fit(exponential,x,morti,p0=[400,0.2],bounds=([0.001,0], [10000,2]),method='trf',
            max_nfev=50000,xtol=1e-15,gtol=1e-15,ftol=1e-15,jac="3-point",loss="linear")
        exp_fit_dead = [exponential(i-inizio_morti,*popt) for i in t]
        r2_exp_dead = r_sqrt(morti,[exponential(i-inizio_morti,*popt) for i in x])
        print("############")

        ################################################

        plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="Epidemia inarrestabile ($R^2="+str(r2_exp)+")$")
        if plot_sir: plt.plot(t,fitted_sir,color="purple",zorder=4,label="S.I.R. Mod. ($R^2="+str(r2_sir)+")$")
        if plot_richards: plt.plot(t,fitted_richards,color="#FFCC00",zorder=3,label="Richards ($R^2="+str(r2_richards)+")$")
        if plot_gompertz: plt.plot(t,fitted_gom,color="#228B22",zorder=3,label="Gompertz ($R^2="+str(r2_gom)+")$")
        if plot_sigmoid: plt.plot(t,fitted_sig,color="red",zorder=3,label="Sigmoide ($R^2="+str(r2_sig)+")$")
        plt.scatter(x,infetti,marker="^",color="black",s=50,zorder=5)
        plt.xlim(0,x[-1]+delta_t2)
        plt.ylim(0,2.5*max(infetti))
        plt.title(str(date)+" in "+regione)
        plt.xlabel("Tempo (Giorni dal 24/02/2020)")
        plt.ylabel("Persone infette")
        plt.legend()
        plt.savefig(directory+"/"+str(date)+"_1.png",dpi=200,bbox_inches='tight')
        plt.clf()

        plt.plot(t,exp_fit_dead,linestyle="-.",zorder=1,label="Epidemia inarrestabile ($R^2="+str(r2_exp_dead)+")$")
        if plot_richards: plt.plot(t,fitted_richards_dead,color="#FFCC00",zorder=3,label="Richards ($R^2="+str(r2_richards_dead)+")$")
        if plot_gompertz: plt.plot(t,fitted_gom_dead,color="#228B22",zorder=3,label="Gompertz ($R^2="+str(r2_gom_dead)+")$")
        if plot_sigmoid: plt.plot(t,fitted_sig_dead,color="red",zorder=3,label="Sigmoide ($R^2="+str(r2_sig_dead)+")$")
        plt.scatter(x,morti,marker="^",color="black",s=50,zorder=5)
        plt.xlim(0,x[-1]+delta_t2)
        plt.ylim(0,2.5*max(morti))
        plt.title(str(date)+" in "+regione)
        plt.xlabel("Tempo (Giorni dal 24/02/2020)")
        plt.ylabel("Persone decedute")
        plt.legend()
        plt.savefig(directory+"/"+str(date)+"_dead_1.png",dpi=200,bbox_inches='tight')
        plt.clf()

        ##################################################################

        plt.plot(t,exp_fit,linestyle="-.",zorder=1,label="$N_{MAX}=\infty$")
        if plot_sir: plt.plot(t,fitted_sir,zorder=3,color="purple",label="$N_{MAX}="+str(int(max_infected_sir))+
            "\pm "+str(round(error_sir/max_infected_sir*100,1))+"\%$")
        if plot_richards: plt.plot(t,fitted_richards,zorder=3,color="#FFCC00",label="$N_{MAX}="+str(int(max_infected_richards))+
            "\pm "+str(round(error_richards/max_infected_richards*100,1))+"\%$")
        if plot_gompertz: plt.plot(t,fitted_gom,zorder=3,color="#228B22",label="$N_{MAX}="+str(int(max_infected_gom))+
            "\pm "+str(round(error_gompertz/max_infected_gom*100,1))+"\%$")
        if plot_sigmoid: plt.plot(t,fitted_sig,zorder=3,color="red",label="$N_{MAX}="+str(int(max_infected_sig))+
        "\pm "+str(round(error_sigmoid/max_infected_sig*100,1))+"\%$")
        plt.scatter(x,infetti,marker="^",color="black",s=40,zorder=4)
        plt.xlim(0,x[-1]+delta_t)
        # plt.ylim(0,max(max(fitted_sig),max(fitted_gom)))
        plt.ylim(0,2.5*max(infetti))
        plt.title(str(date)+" in "+regione)
        plt.xlabel("Tempo (Giorni dal 24/02/2020)")
        plt.ylabel("Persone infette")
        plt.legend()
        plt.savefig(directory+"/"+str(date)+"_2.png",dpi=200,bbox_inches='tight')
        plt.clf()

        plt.plot(t,exp_fit_dead,linestyle="-.",zorder=1,label="$N_{MAX}=\infty$")
        if plot_richards: plt.plot(t,fitted_richards_dead,zorder=3,color="#FFCC00",label="$N_{MAX}="+str(int(max_dead_richards))+
            "\pm "+str(round(error_richards_dead/max_dead_richards*100,1))+"\%$")
        if plot_gompertz: plt.plot(t,fitted_gom_dead,zorder=3,color="#228B22",label="$N_{MAX}="+str(int(max_dead_gom))+
            "\pm "+str(round(error_gompertz_dead/max_dead_gom*100,1))+"\%$")
        if plot_sigmoid: plt.plot(t,fitted_sig_dead,zorder=3,color="red",label="$N_{MAX}="+str(int(max_dead_sig))+
            "\pm "+str(round(error_sigmoid_dead/max_dead_sig*100,1))+"\%$")
        plt.scatter(x,morti,marker="^",color="black",s=40,zorder=4)
        plt.xlim(0,x[-1]+delta_t)
        # plt.ylim(0,max(max(fitted_sig_dead),max(fitted_gom_dead)))
        plt.ylim(0,2.5*max(morti))
        plt.title(str(date)+" in "+regione)
        plt.xlabel("Tempo (Giorni dal 24/02/2020)")
        plt.ylabel("Persone decedute")
        plt.legend()
        plt.savefig(directory+"/"+str(date)+"_dead_2.png",dpi=200,bbox_inches='tight')
        plt.clf()

        ##################################################################

        variazione_infetti = diff(infetti)
        if plot_sigmoid: variazione_teorica_sig = diff(fitted_sig)/(t[1]-t[0])
        if plot_gompertz: variazione_teorica_gom = diff(fitted_gom)/(t[1]-t[0])
        if plot_richards: variazione_teorica_richards = diff(fitted_richards)/(t[1]-t[0])
        if plot_sir: variazione_teorica_sir = diff(fitted_sir)/(t[1]-t[0])

        variazione_morti = diff(morti)
        if plot_sigmoid: variazione_teorica_sig_dead = diff(fitted_sig_dead)/(t[1]-t[0])
        if plot_gompertz: variazione_teorica_gom_dead = diff(fitted_gom_dead)/(t[1]-t[0])
        if plot_richards: variazione_teorica_richards_dead = diff(fitted_richards_dead)/(t[1]-t[0])

        ##################################################################

        plt.plot(x[:-1],variazione_infetti,lw=1,color="blue",linestyle="-.",zorder=1)
        if plot_sir: plt.plot(t[:-1],variazione_teorica_sir,color="purple",zorder=3)
        if plot_richards: plt.plot(t[:-1],variazione_teorica_richards,color="#FFCC00",zorder=2)
        if plot_gompertz: plt.plot(t[:-1],variazione_teorica_gom,color="#228B22",zorder=2)
        if plot_sigmoid: plt.plot(t[:-1],variazione_teorica_sig,color="red",zorder=2)
        plt.scatter(x[:-1],variazione_infetti,marker="^",color="black",s=40,zorder=4)
        plt.xlabel("Tempo (Giorni dal 25/02/2020)")
        plt.ylabel("Nuovi infetti")
        plt.ylim(0,2*max(variazione_infetti))
        plt.title(str(date)+" in "+regione)
        plt.savefig(directory+"/"+"nuovi_infetti.png",dpi=200,bbox_inches='tight')
        plt.clf()

        plt.plot(x[:-1],variazione_morti,lw=1,color="blue",linestyle="-.",zorder=1)
        if plot_richards: plt.plot(t[:-1],variazione_teorica_richards_dead,color="#FFCC00",zorder=2)
        if plot_gompertz: plt.plot(t[:-1],variazione_teorica_gom_dead,color="#228B22",zorder=2)
        if plot_sigmoid: plt.plot(t[:-1],variazione_teorica_sig_dead,color="red",zorder=2)
        plt.scatter(x[:-1],variazione_morti,marker="^",color="black",s=40,zorder=4)
        plt.xlabel("Tempo (Giorni dal 25/02/2020)")
        plt.ylabel("Nuovi deceduti")
        plt.ylim(0,2*max(variazione_morti))
        plt.title(str(date)+" in "+regione)
        plt.savefig(directory+"/"+"nuovi_deceduti.png",dpi=200,bbox_inches='tight')
        plt.clf()

        ##################################################################

        for item in range(len(infetti)):
            infetti[item] = max(infetti[item],1)

        for item in range(len(morti)):
            morti[item] = max(morti[item],1)

        growth_factor = [float(infetti[i+1])/(infetti[i]) for i in range(len(infetti)-1)]
        costante = [1 for i in range(len(growth_factor))]

        growth_factor_dead = [float(morti[i+1])/(morti[i]) for i in range(len(morti)-1)]

        ##################################################################

        plt.plot(x[:-1],costante,color="gray",linestyle="--")
        plt.plot(x[:-1],growth_factor,lw=1,color="blue",linestyle="-.")
        plt.scatter(x[:-1],growth_factor,marker="^",color="black",s=40,zorder=4)
        plt.xlabel("Tempo (Giorni dal 26/02/2020)")
        plt.ylabel("Fattore di crescita")
        plt.title(str(date)+" in "+regione)
        plt.savefig(directory+"/""growth_factor.png",dpi=200,bbox_inches='tight')
        plt.clf()

        plt.plot(x[:-1],costante,color="gray",linestyle="--")
        plt.plot(x[:-1],growth_factor_dead,lw=1,color="blue",linestyle="-.")
        plt.scatter(x[:-1],growth_factor_dead,marker="^",color="black",s=40,zorder=4)
        plt.xlabel("Tempo (Giorni dal 26/02/2020)")
        plt.ylabel("Fattore di crescita (deceduti)")
        plt.title(str(date)+" in "+regione)
        plt.savefig(directory+"/""growth_factor_dead.png",dpi=200,bbox_inches='tight')
        plt.clf()

        ##################################################################

        hist.append(infetti[-1])
        hist2.append(morti[-1])

        ##################################################################
        if regione == "P.A. Bolzano":
            regione = "P.A. Trento"

        url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province-latest.csv"
        df = pd.read_csv(url,skiprows=1,names=["data","stato","codice_regione",
        "denominazione_regione","codice_provincia","denominazione_provincia","sigla_provincia","lat","long","totale_casi","note_it","note_en"])
        df = df.replace("P.A. Bolzano","P.A. Trento").replace("Emilia-Romagna","Emilia Romagna")
        df = df[df["denominazione_regione"] == regione]
        df = df[df["denominazione_provincia"] != "In fase di definizione/aggiornamento"]

        shapefile = 'map/ITA_adm2.shp'
        gdf = gpd.read_file(shapefile).replace("Apulia","Puglia").replace("Emilia-Romagna","Emilia Romagna")
        gdf = gdf.replace("Friuli-Venezia Giulia","Friuli Venezia Giulia").replace("Sicily","Sicilia").replace("Trentino-Alto Adige","P.A. Trento")

        gdf = gdf.replace("Padua","Padova").replace("Florence","Firenze").replace("Syracuse","Siracusa")
        gdf = gdf.replace("Carbonia-Iglesias","Sud Sardegna").replace("Medio Campidano","Sud Sardegna").replace("Ogliastra","Sud Sardegna").replace("Olbia-Tempio","Sud Sardegna")
        gdf = gdf.replace("Mantua","Mantova").replace("Monza and Brianza","Monza e della Brianza").replace("Forli' - Cesena","Forlì-Cesena")
        gdf = gdf.replace("Reggio Nell'Emilia","Reggio nell'Emilia").replace("Reggio Di Calabria","Reggio di Calabria")

        gdf = gdf[gdf["NAME_1"] == regione]

        merged = gdf.merge(df, left_on = "NAME_2", right_on = "denominazione_provincia")
        # print(merged)

        merged.plot(column="totale_casi",cmap="coolwarm",legend=True, scheme="NaturalBreaks", k=8,
        edgecolor="k", linewidth=0.05, legend_kwds={"loc": "center left", "bbox_to_anchor": (1,0.5)})
        plt.title(str(date)+" - Totale Contagiati\n"+regione)
        plt.axis('off')
        plt.savefig(directory+"/"+str(date)+"_map.png",dpi=300,bbox_inches='tight')
        plt.clf()

        it_ab += 1

    for i in range(len(regioni)):
        plt.bar(i,hist[i])
    plt.ylabel("Totale infetti")
    plt.xticks(range(len(regioni)),regioni,rotation="vertical")
    plt.title(str(date)+" in Italia")
    plt.savefig("img_italia/"+str(date)+"_regioni.png",dpi=200,bbox_inches='tight')
    plt.close("all")

    for i in range(len(regioni)):
        plt.bar(i,hist2[i])
    plt.ylabel("Totale deceduti")
    plt.xticks(range(len(regioni)),regioni,rotation="vertical")
    plt.title(str(date)+" in Italia")
    plt.savefig("img_italia/"+str(date)+"_regioni_dead.png",dpi=200,bbox_inches='tight')
    plt.clf()

if plot_map:
    url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-latest.csv"
    df = pd.read_csv(url,skiprows=1,names=["data","stato","codice_regione",
    "denominazione_regione","lat","long","ricoverati_con_sintomi",
    "terapia_intensiva","totale_ospedalizzati","isolamento_domiciliare",
    "totale_positivi","variazione_totale_positivi","nuovi_positivi",
    "dimessi_guariti","deceduti","totale_casi","tamponi","note_it","note_en"])

    df.at[17, "totale_casi"] += df["totale_casi"][2]
    df = df.drop(df.index[2])

    shapefile = 'map/ITA_adm1.shp'
    gdf = gpd.read_file(shapefile)

    ids = [13,16,17,18,15,8,6,12,7,3,11,14,1,20,19,9,4,10,2,5]

    for i in range(len(ids)):
        gdf.at[i, "ID_1"] = ids[i]

    merged = gdf.merge(df, left_on = "ID_1", right_on = "codice_regione")

    merged.plot(column="totale_casi",cmap="coolwarm",legend=True, scheme="NaturalBreaks", k=8,
    edgecolor="k", linewidth=0.05, legend_kwds={"loc": "center left", "bbox_to_anchor": (1,0.5)})
    plt.title(str(date)+" - Totale Contagiati")
    plt.axis('off')
    plt.savefig("img_italia/"+str(date)+"_map.png",dpi=300,bbox_inches='tight')
    plt.clf()
