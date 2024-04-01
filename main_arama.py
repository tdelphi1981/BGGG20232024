import math
import pickle
from bisect import bisect_left
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, csr_matrix

from onisleme import onisle


def index(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return None


def main():
    deney_klasor = Path("deneyler") / "deney1"
    sozluk_dosyasi = deney_klasor / "sozluk.szl"
    with sozluk_dosyasi.open("rb") as dosya:
        sozluk = pickle.load(dosya)
        N = pickle.load(dosya)
    print(f"Toplam makale sayısı: {N}")
    print(f"Toplam kelime sayısı: {len(sozluk)}")
    tfidf_dosyasi = deney_klasor / "tfidf.idx"
    with tfidf_dosyasi.open("rb") as dosya:
        tdm = load_npz(dosya)
    print(f"Terim-Doküman Matrisi Doluluk Oranı: {math.ceil(10000 * tdm.nnz / (tdm.shape[0] * tdm.shape[1])) / 100}")
    print("Çıkmak için \\c giriniz.")
    while True:
        metin = input("Arama Metni Girin > ")
        if metin == '\\c':
            break
        print(f"Aradığınız metin : {metin}")
        arama_metin_parcalari = onisle(metin)
        arama_sozluk_indisleri = [index(sozluk, parca) for parca in arama_metin_parcalari]
        temiz_sozluk_indisleri = [ind for ind in arama_sozluk_indisleri if ind is not None]
        sorgu_vektoru = csr_matrix((1, len(sozluk)))
        sorgu_vektoru[0, temiz_sozluk_indisleri] = 1 / math.sqrt(len(temiz_sozluk_indisleri))
        cosinus_benzerlikleri = tdm.dot(sorgu_vektoru.T).toarray().flatten()
        sira_numaralari = np.flip(cosinus_benzerlikleri.argsort())
        print(sira_numaralari[:10])



if __name__ == '__main__':
    main()
