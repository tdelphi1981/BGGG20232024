# Wikipedi dosyasındaki text bilgisine erişilmeli!
from datetime import datetime
from json import loads

import numpy as np

# DENEME 2
# JSON görünüyor, Python JSON kütüphanesi ile bilgi almayı deneyelim

# loads bir metin bilgisini Python nesnesine dönüştürür
print("1- Veri Yükleme")
genel_baslangic = datetime.now()
baslangic = genel_baslangic
orneklem = []

with open("veri/wikipedia_10000.json", "r") as dosya:
    for satir in dosya:
        ornek = loads(satir)
        orneklem.append(ornek)

print(f"Örneklem sayısı {len(orneklem)} tamamlanma süresi: {datetime.now() - baslangic}")

# Doküman Tanımı
print("1.1- Doküman Ayıklama")
baslangic = datetime.now()
orneklem = [ornek["text"] for ornek in orneklem]
print(f"Ayıklanan doküman sayısı {len(orneklem)} tamamlanma süresi: {datetime.now() - baslangic}")

# Metin Önişleme


print("2- Ön işleme")

print("2.1- Karakter Önişleme")
baslangic = datetime.now()
# Karakter dönüşümü
# Python str-> lower(): küçük harfe çevirir, upper(): büyük harfe çevirir

# for i, metin in enumerate(orneklem):
#     orneklem[i] = metin.lower()
orneklem = [metin.lower() for metin in orneklem]

import unicodedata
import re

# Unicode Normalizasyonu
# ö: 1-ö, 2-o ..
orneklem = [unicodedata.normalize('NFKD', metin) for metin in orneklem]

# Karakter Eliminasyonu
# Seçilen Karakter Kategorileri: Ll, Nd, Zs

secilen_kategoriler = ['Ll', 'Nd', 'Zs']

for i, ornek in enumerate(orneklem):
    kategoriler = [unicodedata.category(karakter) for karakter in ornek]
    yeni_metin = "".join([ornek[j] if kategoriler[j] in secilen_kategoriler and kategoriler[j] != 'Zs'
                          else ' ' for j in range(len(ornek))])
    # yeni_metin = []
    # for j, kategori in enumerate(kategoriler):
    #     if kategori in secilen_kategoriler:
    #         if kategori == 'Zs':
    #             yeni_metin.append(' ')
    #         else:
    #             yeni_metin.append(ornek[j])
    #     else:
    #         yeni_metin.append(' ')
    # yeni_metin = "".join(yeni_metin)  # listeyi metne geri çevir
    # Yeni problem: Çoklu boşluk problemi

    yeni_metin = re.sub(' +', ' ', yeni_metin)
    # while True:
    #     uzunluk = len(yeni_metin)
    #     yeni_metin = yeni_metin.replace("  ", " ")
    #     if len(yeni_metin) == uzunluk:
    #         break
    # En iyi alternatif REGEX
    orneklem[i] = yeni_metin.strip()

print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

print("2.2- Metin Parçalama")
baslangic = datetime.now()

# for i, ornek in enumerate(orneklem):
#    orneklem[i] = ornek.split(' ')
orneklem = [ornek.split(' ') for ornek in orneklem]
print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

print("2.3- Metin Parçası Önişleme")
baslangic = datetime.now()

import nltk

budayici = nltk.stem.SnowballStemmer('english')

zamirler = nltk.corpus.stopwords.words('english')

# for ornek in orneklem:
#    zamirsiz_ornek = [parca for parca in ornek if parca not in zamirler]
#    budanmis_ornek = [budayici.stem(parca) for parca in zamirsiz_ornek]

# for i,ornek in enumerate(orneklem):
#    orneklem[i] = [budayici.stem(parca) for parca in ornek if parca not in zamirler]

orneklem = [[budayici.stem(parca) for parca in ornek if parca not in zamirler] for ornek in orneklem]
print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

print("3- Sayısallaştırma")

print("3.1- Sözlük Oluşturma")
baslangic = datetime.now()
sozluk = set()
for ornek in orneklem:
    # for kelime in ornek:
    #     if kelime not in sozluk:
    #         sozluk.append(kelime)
    sozluk.update(ornek)

sozluk = list(sozluk)

sozluk.sort()
print(f"Sözlük boyutu {len(sozluk)} tamamlanma süresi: {datetime.now() - baslangic}")

print("3.2- Sayısallaştırma")
baslangic = datetime.now()

# Kelimeden Sıra Numarasına gitme
import bisect

sayisal_orneklem = []

for ornek in orneklem:
    sayisal_ornek = [bisect.bisect_left(sozluk, kelime) for kelime in ornek]
    sayisal_orneklem.append(sayisal_ornek)

print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

print("3.2.1- Doküman Frekansaları Hesaplanıyor")
baslangic = datetime.now()
from collections import Counter

frekans_orneklem = [Counter(dokuman) for dokuman in sayisal_orneklem]
print(f"Tamamlanma süresi: {datetime.now() - baslangic}")
baslangic = datetime.now()
print("3.2.1.1- Terim Frekansları Hesaplanıyor")

N = len(orneklem)
M = len(sozluk)

tdm = np.zeros((N, M))

for i, frekanslar in enumerate(frekans_orneklem):
    avgfik = sum(frekanslar.values()) / len(frekanslar)
    katsayi = 1 / (1 + np.log10(avgfik))
    for j, fik in frekanslar.items():
        tdm[i, j] = (1 + np.log10(fik)) * katsayi
print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

baslangic = datetime.now()
print("3.2.1.1- Doküman Frekansları Hesaplanıyor")

A = tdm > 0

df = A.sum(axis=0)

idf = np.log10(N / df)

print(f"Tamamlanma süresi: {datetime.now() - baslangic}")

tfidf = tdm * idf

baslangic = datetime.now()
print("3.2.2- Doküman Vektörü Normalizasyonu")

dokuman_uzunluklari = (tfidf ** 2).sum(axis=1)

for i in range(N):
    tfidf[i, :] = tfidf[i, :] / np.sqrt(dokuman_uzunluklari[i])

print("3.3- Sparse Matrise çevirme")

from scipy.sparse import csr_matrix

tfidf_sparse = csr_matrix(tfidf)

# tfidf_sparse.indptr # RI
# tfidf_sparse.indices # CI
# tfidf_sparse.data # V

d5 = tfidf_sparse[5, :]

d5 = tfidf_sparse.data[tfidf_sparse.indptr[5]:tfidf_sparse.indptr[6]]

print(f"Genel Tamamlanma süresi : {datetime.now() - genel_baslangic}")
