import bisect
import re
import unicodedata
from collections import Counter
from datetime import datetime
from json import loads

import nltk
import numpy as np
from scipy.sparse import lil_matrix

budayici = nltk.stem.SnowballStemmer('english')
zamirler = nltk.corpus.stopwords.words('english')


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


def sozluk_olustur(dosyaadi: str = "veri/wikipedia_10000.json"):
    sozluk = set()
    N = 0
    with open(dosyaadi, "r") as dosya:
        for satir in dosya:
            ornek = loads(satir)
            dokuman = ornek["text"]

            metin_parcalari = onisle(dokuman)

            sozluk.update(metin_parcalari)
            N += 1
            print(f"{N}. doküman tamamlandı...")

    sozluk = list(sozluk)
    sozluk.sort()

    return sozluk, N


def tfidf(sozluk: list[str], N: int, dosyaadi: str = "veri/wikipedia_10000.json"):
    tfidf = lil_matrix((N, len(sozluk)))

    with open(dosyaadi, "r") as dosya:
        for i, satir in enumerate(dosya):
            ornek = loads(satir)
            dokuman = ornek["text"]

            metin_parcalari = onisle(dokuman)

            sozluk_sira_nolari = [bisect.bisect_left(sozluk, kelime) for kelime in metin_parcalari]

            frekanslar = Counter(sozluk_sira_nolari)

            sutun_sira_nolari = np.array(list(frekanslar.keys()))
            frekans_degerleri = np.array(list(frekanslar.values()))

            tfidf[i, sutun_sira_nolari] = frekans_degerleri

            print(f"{i + 1}. doküman dizinlendi...")

    tfidf = tfidf.tocsr()

    return tfidf


def main(dosyaadi: str = "veri/wikipedia_10000.json"):
    baslangic = datetime.now()
    sozluk, N = sozluk_olustur(dosyaadi)
    tdm = tfidf(sozluk, N, dosyaadi)
    print(f"Tamamlanma süresi: {datetime.now() - baslangic}")


if __name__ == '__main__':
    main("veri/wikipedia_1000.json")
