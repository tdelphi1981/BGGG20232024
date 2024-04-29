import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, csr_matrix


def ilklendir(V: csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    # V -> m,n
    # W -> m,k
    # H -> k,n
    max_v = np.max(V.data)
    min_v = np.min(V.data)
    m, n = V.shape
    w = np.random.uniform(low=min_v, high=max_v, size=(m, k))
    h = np.random.uniform(low=min_v, high=max_v, size=(k, n))

    return w, h


def optimizasyon(V: csr_matrix, w: np.ndarray, h: np.ndarray):
    epsilon = 1e-2
    i = 0
    while True:
        hn1 = h * ((w.T @ V) / (w.T @ w @ h))
        wn1 = w * ((V @ hn1.T) / (w @ hn1 @ hn1.T))
        eps_h = np.linalg.norm(h - hn1, 2)
        eps_w = np.linalg.norm(w - wn1, 2)
        h = hn1
        w = wn1
        if eps_w < epsilon and eps_h < epsilon:
            break
        i += 1
        print(f"İterasyon {i} tamamlandı, hata: H: {eps_h}, W: {eps_w}")
    return w, h


def nmf(V: csr_matrix, k: int):
    w0, h0 = ilklendir(V, k)
    w, h = optimizasyon(V, w0, h0)
    return w, h


def main(deney_adi: str):
    deneyKlasoru = Path('deneyler') / deney_adi

    sozlukDosyasi = deneyKlasoru / 'sozluk.szl'
    tfidfDosyasi = deneyKlasoru / 'tfidf.idx'

    with sozlukDosyasi.open("rb") as f:
        sozluk = pickle.load(f)

    with tfidfDosyasi.open("rb") as f:
        tfidf = load_npz(f)

    konu_sayisi = 5

    w, h = nmf(tfidf, konu_sayisi)

    print("NMF Tamamlandı...")



if __name__ == '__main__':
    main('deney2')
