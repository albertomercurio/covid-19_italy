# covid-19_italy
Analisi dati del COVID 19 in Italia


Utilizzo i dati forniti dalla protezione civile:
https://github.com/pcm-dpc/COVID-19

Il file covid19_italy.py è il programma che genera otto immagini grafici: "data"_1.png e "data"_2.png, "data"_dead_1.png e "data"_dead_2.png, nuovi_infetti.png, nuovi_deceduti.png, growth_fanctor.png e growth_fanctor_dead.png.

La prima e la terza immagine, mostrano (rispettivamente per i contagiati ed i deceduti) il fit dei dati fino al giorno specificato.
La seconda e la quarta, mostrano (rispettivamente per i contagiati ed i deceduti) lo stesso grafico ma sul lungo periodo.
La quinta mostra i nuovi infetti al giorno, definita come la differenza tra gli infetti totali al giorno X meno gli infetti totali al giorno X-1.
La sesta mostra i nuovi deceduti al giorno, definita come la differenza tra i deceduti totali al giorno X meno i deceduti totali al giorno X-1.
La settima mostra il fattore di crescita, definito come il rapporto tra gli infetti al giorno X e gli infetti al giorno X-1.
L0ottava mostra il fattore di crescita, definito come il rapporto tra i deceduti al giorno X e i deceduti al giorno X-1.

In più è presente una mappa per ogni regione raffigurante i casi totali in ogni provincia.

I fit sono stati eseguiti considerando diverse tipologie di funzioni: Sigmoide, Gompertz, Richards e S.I.R. Modificata.
L'ultima è una modifica del modello S.I.R. in cui è stata aggiunta l'azione di lockdown, per maggiori chiarimenti vedere l'articolo originale:
https://www.sciencedirect.com/science/article/pii/S0960077920301636
