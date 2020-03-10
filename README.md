# covid-19_italy
Analisi dati del COVID 19 in Italia


Utilizzo i dati forniti dalla protezione civile:
https://github.com/pcm-dpc/COVID-19

Il file covid19_italy.py è il programma che genera quattro immagini grafici: "data"_1.png e "data"_2.png, nuovi_infetti.png e growth_fanctor.png.

La prima immagine, mostra il fit dei dati fino al giorno specificato.
La seconda, mostra lo stesso grafico ma sul lungo periodo.
La terza mostra i nuovi infetti al giorno, definita come la differenza tra gli infetti totali al giorno X meno gli infetti totali al giorno X-1.
La quarta mostra il fattore di crescita, definito come il rapporto tra i nuovi infetti al giorno X e i nuovi infetti al giorno X-1.

Il secondo grafico mostra il comportamento teorico dell'epidemia a distanze temporali maggiori.
