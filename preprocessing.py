import wfdb
import numpy as np
from scipy.signal import find_peaks
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
        offset = int(fs * 120)  # offset di 2 minuti come descritto da paper
        start_offset = start + offset   
        for j in range(start_offset, end - window_size, step):  #suddivisione del periodo del segnale lo faccio secondo finestre di 60 secondi.
            window = segnale[j:j + window_size, :] #estraggo la finestra del segnale
            ALL_X.append(window) 
            ALL_Y.append(label) 

print(f"Totale finestre: {len(ALL_X)}")
print(f"Distribuzione: no_stress={ALL_Y.count(0)}, stress={ALL_Y.count(1)}")

#######################INIZIO FILTRAGGIO################

#def filtraggio segnale ecg, tramite giustamente il segnale, la freq di campionamento e frequenza di taglio (0.5 per eliminare la componente lenta)
def filter_ecg(signal, fs, cutoff=0.5): 
    N = len(signal) #lunghezza del segnale

    X = np.fft.fft(signal) #passo dal dominio del tempo a quello delle frequenze.
    freqs = np.fft.fftfreq(N, d=1/fs) #creo array per indicare a quale frequenza corrisponde ogni punto di fft.
    X[np.abs(freqs) < cutoff] = 0 #se la frequenza è al di sotto della frequenza di taglio, la azzero.
    return np.real(np.fft.ifft(X)) #restituisco il segnale, riportato però nel periodo del tempo.

def filter_gsr(signal): 
    N = len(signal) #numero campioni del segnale.
    X = np.fft.fft(signal) #porto il segnale nel dominio delle frequenze.
    X[0] = 0 #X[0] è la frequenza 0 ovvero la componente costante del segnale, dunque la elimino
    return np.real(np.fft.ifft(X)) 

ECG_COL = 0 #############
EMG_COL = 1 #############
FGSR_COL = 2  ############# modifica: aggiunta dei parametri mancanti perché necessari nell'estrazione delle feature
HGSR_COL = 3 #############
RESP_COL = 6 #############


#escludo HR (non previsto nel paper)
#escludo anche EMG 

ALL_X_filtered = []
#filtro vero e proprio.
for window in ALL_X: 
    w = window.copy()
    w[:, ECG_COL] = filter_ecg(w[:, ECG_COL], fs)
    w[:, FGSR_COL] = filter_gsr(w[:, FGSR_COL])
    w[:, HGSR_COL] = filter_gsr(w[:, HGSR_COL])
    ALL_X_filtered.append(w)

print(f"Finestre filtrate: {len(ALL_X_filtered)}")




#funzione di estrazione picchi
def extract_peak_features(signal, fs, width=5, prominence=0.1):
    peaks, _ = find_peaks(signal, width=width, prominence=prominence) #estraggo i picchi dal segnale in base ai parametri di promimance (picchi più evidenti) e width per evitare picchi troppo stretti.
    n_peaks = len(peaks) #conto il numero di picchi
    if n_peaks > 1: #se ho almeno due picchi (non posso calcolare la distanza se non ne ho almeno due)
        intervals = np.diff(peaks) / fs #intervallo tra picchi
        mean_interval = np.mean(intervals) #intervallo medio
        amp_diffs = np.diff(signal[peaks]) #differenza di ampiezza tra i picchi
        mean_diff = np.mean(np.abs(amp_diffs)) #media delle ampiezze.
    else: #se non ho picchi giustamento tutti i valori verrano settati a zero.
        mean_interval = 0
        mean_diff = 0
    return n_peaks, mean_interval, mean_diff 

#funzione per il calcolo della intensità media efficace 
def extract_rms(signal): 
    return np.sqrt(np.mean(signal**2))

FEATURES = [] #lista delle features.
for window in ALL_X_filtered:
    #prendo tutti i segnali filtrati
    ecg = window[:, ECG_COL] 
    fgsr = window[:, FGSR_COL]
    hgsr = window[:, HGSR_COL]
    emg = window[:, EMG_COL]
    resp = window[:, RESP_COL]
    #estraggo le features di ogni segnale
    peaks_ecg, mean_rr, _ = extract_peak_features(ecg, fs)
    peaks_fgsr, mean_int_f, mean_diff_f = extract_peak_features(fgsr, fs)
    peaks_hgsr, mean_int_h, mean_diff_h = extract_peak_features(hgsr, fs)
    rms_emg = extract_rms(emg)
    mean_resp = np.mean(resp)
    #le salvo
    FEATURES.append([
        peaks_ecg, mean_rr,
        peaks_fgsr, mean_int_f, mean_diff_f,
        peaks_hgsr, mean_int_h, mean_diff_h,
        rms_emg, mean_resp
    ])
#le trasformo in un array numpy per poterle salvare dopo più facilmente.
FEATURES = np.array(FEATURES)
Y = np.array(ALL_Y)

print(f"Shape delle feature: {FEATURES.shape}") 
np.save("FEATURES.npy", FEATURES)
np.save("Y.npy", Y)