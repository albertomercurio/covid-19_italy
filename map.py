import datetime
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

date = datetime.date.today()

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-latest.csv"
df = pd.read_csv(url,skiprows=1,names=["data","stato","codice_regione",
"denominazione_regione","lat","long","ricoverati_con_sintomi","terapia_intensiva",
"totale_ospedalizzati","isolamento_domiciliare","totale_attualmente_positivi","nuovi_attualmente_positivi",
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

print("#######################################################")

#######################################################
#######################################################
#######################################################

regioni = ["Abruzzo","Basilicata","Calabria","Campania","Emilia Romagna","Friuli Venezia Giulia",
"Lazio","Liguria","Lombardia","Marche","Molise","Piemonte","Puglia","Sardegna","Sicilia","Toscana","P.A. Trento",
"Umbria","Valle d'Aosta","Veneto"]

for regione in range(1):
    regione = "Sardegna"
    directory = ("img_"+regione).replace(" ","").lower()

    url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province-latest.csv"
    df = pd.read_csv(url,skiprows=1,names=["data","stato","codice_regione",
    "denominazione_regione","codice_provincia","denominazione_provincia","sigla_provincia","lat","long","totale_casi","note_it","note_en"])
    df = df.replace("P.A. Bolzano","P.A. Trento")
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
    print(merged)

    merged.plot(column="totale_casi",cmap="coolwarm",legend=True, scheme="NaturalBreaks", k=8,
    edgecolor="k", linewidth=0.05, legend_kwds={"loc": "center left", "bbox_to_anchor": (1,0.5)})
    plt.title(str(date)+" - Totale Contagiati\n"+regione)
    plt.axis('off')
    plt.savefig(directory+"/"+str(date)+"_map.png",dpi=300,bbox_inches='tight')
    plt.clf()
