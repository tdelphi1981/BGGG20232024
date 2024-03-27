import argparse
import bisect
import multiprocessing
import os
import pickle
import re
import unicodedata
from collections import Counter
from datetime import datetime
from json import loads
from pathlib import Path

import nltk
import numpy as np
import scipy.sparse
from scipy.sparse import lil_matrix, save_npz, load_npz

budayici = nltk.stem.SnowballStemmer('english')
zamirler = nltk.corpus.stopwords.words('english')


def dosyaOku(dosyaAdi: str, blokBoyutu: int = 1):
    blok = []
    with open(dosyaAdi, "r") as dosya:
        for satir in dosya:
            jsn = loads(satir)
            blok.append(jsn["text"])
            if len(blok) == blokBoyutu:
                yield blok
                blok = []
        if len(blok) > 0:
            yield blok


def onisle(metin: str) -> list[str]:
    metin = metin.lower()
    metin = unicodedata.normalize('NFKD', metin)
    secilen_kategoriler = ['Ll', 'Nd', 'Zs']
    kategoriler = [unicodedata.category(karakter) for karakter in metin]
    yeni_metin = "".join([metin[j] if kategoriler[j] in secilen_kategoriler and kategoriler[j] != 'Zs'
                          else ' ' for j in range(len(metin))])
    metin = re.sub(' +', ' ', yeni_metin)

    metin = metin.strip()

    metin = metin.split()

    metin = [budayici.stem(parca) for parca in metin if parca not in zamirler]

    return metin


def sozluk_olustur(dosyaadi: str, process_sayisi: int = 10):
    baslangic = datetime.now()
    print("Sözlük oluşturuluyor...")
    sozluk = set()
    N = 0
    with multiprocessing.Pool(processes=process_sayisi) as p:
        for blok in dosyaOku(dosyaadi, 100):
            metin_parcalari = [p.apply_async(onisle, (dokuman,)).get()
                               for dokuman in blok]

            for parcalar in metin_parcalari:
                sozluk.update(parcalar)
                N += 1
                print(f"{N}. doküman tamamlandı...", end='\r')

    sozluk = list(sozluk)
    sozluk.sort()
    print(f"\nSözlük Oluşturuldu kelime sayısı {len(sozluk)}, tamamlanma süresi {datetime.now() - baslangic}")
    return sozluk, N


def tfidf(sozluk: list[str], N: int, dosyaadi: str):
    baslangic = datetime.now()
    print("TF-IDF oluşturuluyor...")
    tfidf = lil_matrix((N, len(sozluk)))
    i = 0
    for blok in dosyaOku(dosyaadi, 100):
        blok_metin_parcalari = [onisle(metin) for metin in blok]

        sozluk_sira_nolari = [[bisect.bisect_left(sozluk, kelime) for kelime in metin_parcalari]
                              for metin_parcalari in blok_metin_parcalari]

        blok_frekanslar = [Counter(ssn) for ssn in sozluk_sira_nolari]

        sutun_sira_nolari = [np.array(list(bf.keys())) for bf in blok_frekanslar]
        frekans_degerleri = [np.array(list(bf.values())) for bf in blok_frekanslar]

        for sut, frekans in zip(sutun_sira_nolari, frekans_degerleri):
            tfidf[i, sut] = frekans
            i += 1

        print(f"{i}. doküman dizinlendi...", end='\r')

    tfidf = tfidf.tocsr()

    satir_toplam = np.asarray(tfidf.sum(axis=1).flatten()).reshape(-1)
    satir_es = tfidf.indptr[1:] - tfidf.indptr[:-1]
    satir_ort = satir_toplam / satir_es
    satir_ort = 1 + np.log10(satir_ort)

    tfidf.data = 1 + np.log10(tfidf.data)

    for i in range(N):
        tfidf[i, :] /= satir_ort[i]

    print("IDF hesaplanıyor")

    tfidf = tfidf.tocsc()

    sutun_es = tfidf.indptr[1:] - tfidf.indptr[:-1]

    idf = np.log10((N - sutun_es) / sutun_es)

    print("TF*IDF Hesaplanıyor")

    tfidf = tfidf.tocsr()

    for i in range(N):
        tfidf[i, :] *= idf[i]

    print("Normalizasyon yapılıyor")

    # tfsq = tfidf * tfidf
    # uzunluk = np.sqrt(tfsq.sum(axis=1).flatten())
    #
    # for i in range(N):
    #     tfidf[i, :] /= uzunluk[i]

    for i in range(N):
        satir = tfidf.data[tfidf.indptr[i]:tfidf.indptr[i + 1]]
        tfidf[i, :] /= np.sqrt(np.sum(satir * satir))

    print(f"\nTF-IDF oluşuturuldu, tamamlanma süresi {datetime.now() - baslangic}")
    return tfidf


def main(parametreler):
    deney_ana_klasor = Path("deneyler")

    deney_klasor = deney_ana_klasor / parametreler.deneyadi

    if not deney_klasor.exists():
        deney_klasor.mkdir(parents=True)

    baslangic = datetime.now()

    sozluk_dosyasi = deney_klasor / "sozluk.szl"
    if not sozluk_dosyasi.exists():
        sozluk, N = sozluk_olustur(parametreler.dosyaadi,
                                   process_sayisi=parametreler.process_sayisi)
        with sozluk_dosyasi.open("wb") as dosya:
            pickle.dump(sozluk, dosya)
            pickle.dump(N, dosya)
    else:
        with sozluk_dosyasi.open("rb") as dosya:
            sozluk = pickle.load(dosya)
            N = pickle.load(dosya)

    tfidf_dosyasi = deney_klasor / "tfidf.idx"
    if not tfidf_dosyasi.exists():
        tdm = tfidf(sozluk, N, parametreler.dosyaadi)
        with tfidf_dosyasi.open("wb") as dosya:
            save_npz(dosya, tdm, compressed=True)
    else:
        with tfidf_dosyasi.open("rb") as dosya:
            tdm = load_npz(dosya)

    print(f"Tamamlanma süresi: {datetime.now() - baslangic}")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(prog="NLP Dizinleyici",
                                         description="Bu program cverilen bir Wikipedia JSON dosyasını klasik NLP "
                                                     "Yöntemleri iler dizinler.")
    arg_parser.add_argument("dosyaadi", help="Dizinlenecek dosya adı")
    arg_parser.add_argument("-p", "--process-sayisi", dest="process_sayisi",
                            type=int, default=os.cpu_count()//2, help="Önişlemede kullanılacak Process sayisi")
    arg_parser.add_argument("-d", "--deney-adi", dest="deneyadi",
                            type=str, default="deney1", help="Oluşturulacak deney klasörü")
    args = arg_parser.parse_args()

    main(parametreler=args)
