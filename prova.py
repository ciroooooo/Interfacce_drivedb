import wfdb
import numpy as np


#dizionario con i periodi di inizio e fine di ogni periodo dell'esperimento (presi dal paper)
period_bounds = {
    "05": [515, 14586, 29468, 36666, 42306, 49337, 63246, 77922],
    "06": [1502, 15498, 28977, 35786, 41862, 48965, 60399, 74394],
    "07": [1513, 15502, 30592, 40781, 49922, 57026, 66466, 80440],
    "08": [528, 14480, 25930, 32654, 41495, 48601, 61091, 75108],
    "09": [892, 15456, 33319, 41199, 46038, 52607, 64895, None],
    "10": [2410, 16395, 30626, 38676, 43575, 50126, 61340, 75093],
    "11": [1489, 15454, 30159, 37070, 43715, 50192, 61091, 75033],
    "12": [4825, 18787, 31260, 38288, 44330, 51830, 62693, 76648],
    "15": [345, 14296, 25957, 32688, 38260, 44598, 55866, 69815],
    "16": [792, 14750, 29739, 36380, 41145, 47476, 60418, None],
}


# nomi e label (0 = no stress, 1 = stress) — "return" viene escluso
period_names = ["rest1", "city1", "hwy1", "return", "hwy2", "city2", "rest2"]
period_labels = [0, 1, 1, None, 1, 1, 0]  # None = periodo da scartare

#lista con i numeri dei driver (per iterazione)
drivers = ["05", "06", "07", "08", "09", "10", "11", "12", "15", "16"]

ALL_X = [] #finestre del segnale 
ALL_Y = [] #classe rispettiva

window_size_sec = 60  #uso finestre da 60 secondi (per replicare il paper)
step_sec = window_size_sec

for driver in drivers: 
    record = wfdb.rdrecord(f"drivedb/drive{driver}") #prelevo il record
    segnale = record.p_signal #estraggo i dati fisiologi del guidatore
    fs = record.fs #frequenza di campionamento.

    bounds = period_bounds[driver]  #prendo la lista dei periodi del guidatore corrente.
    window_size = int(fs * window_size_sec) #dimensione della finestra di campionamento (in campioni)
    step = int(fs * step_sec) 

    for i in range(len(period_names)): #ciclo per ogni periodo (rest1, city1, hwy1, return, hwy2, city2, rest2)
        start = bounds[i]   #inizio del periodo
        end = bounds[i + 1] #fine del periodo
        label = period_labels[i] #label del periodo

        if start is None or end is None or label is None: #controllo per saltare i periodi mancanti e quello di return che è escluso
            continue  

        for j in range(start, end - window_size, step):  #suddivisione del periodo del segnale lo faccio secondo finestre di 60 secondi.
            window = segnale[j:j + window_size, :] #estraggo la finestra del segnale
            ALL_X.append(window) #salvo
            ALL_Y.append(label) #salvo

print(f"Totale finestre: {len(ALL_X)}")
print(f"Distribuzione: no_stress={ALL_Y.count(0)}, stress={ALL_Y.count(1)}")