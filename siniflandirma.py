import math
from collections import Counter
from pathlib import Path
from pprint import pprint

import numpy as np
import random as rnd
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix
from tokenizers import Tokenizer

kategoriler = ["magazin", "saglik"]


def veri_hazirlama() -> list[dict[str, str]]:
    # etiketli veri hazırlanmalı
    veri_klasoru = Path("veri/42bin_haber/news")

    veriler = []

    for kategori in kategoriler:
        kategori_klasoru = veri_klasoru / kategori
        for dosya in kategori_klasoru.glob("*.txt"):
            with dosya.open("r") as f:
                dosya_icerigi = f.readlines()
            # Enter karakterinden (\n) kurtulalım
            dosya_icerigi = [satir.strip() for satir in dosya_icerigi]
            dosya_icerigi = " ".join(dosya_icerigi)

            veriler.append({
                "kategori": kategori,
                "dosya_icerigi": dosya_icerigi,
            })
    return veriler


def sayisallastir(veriler: list[dict[str, str]]) -> tuple[csc_matrix, list[int]]:
    metinler = [veri["dosya_icerigi"] for veri in veriler]
    siniflar = [kategoriler.index(veri["kategori"]) for veri in veriler]

    tokenizer = Tokenizer.from_file("tokenizer.json")

    sayisal_metinler = []

    for metin in metinler:
        sayisal_metinler.append(tokenizer.encode(metin).ids)

    frekanslar = []
    for sayisal_metin in sayisal_metinler:
        frekanslar.append(Counter(sayisal_metin))

    tfidf = lil_matrix((len(veriler), tokenizer.get_vocab_size()))

    for i, frekans in enumerate(frekanslar):
        sutun_sira_nolari = np.array(list(frekans.keys()))
        frekans_degerleri = np.array(list(frekans.values()))

        for sut, frekans in zip(sutun_sira_nolari, frekans_degerleri):
            tfidf[i, sut] = frekans

    tfidf = tfidf.tocsc()

    return tfidf, siniflar


def veriyi_bol(tf: csc_matrix, siniflar: list[int], egitim_orani: float) -> tuple[
    dict[int, csc_matrix], dict[int, csc_matrix]]:
    sinif_frekanslari = Counter(siniflar)
    egitim_miktari = {sinif: int(sinif_frekanslari[sinif] * egitim_orani)
                      for sinif in sinif_frekanslari}

    ornek_indexleri = {
        sinif: np.where(np.array(siniflar) == sinif)[0].tolist() for sinif in sinif_frekanslari
    }

    egitim_kumesi_indexleri = {
        sinif: rnd.sample(ornek_indexleri[sinif], egitim_miktari[sinif]) for sinif in sinif_frekanslari
    }

    test_kumesi_indexleri = {
        sinif: [i for i in ornek_indexleri[sinif] if i not in egitim_kumesi_indexleri[sinif]]
        for sinif in sinif_frekanslari
    }

    egitim_kumesi = {
        sinif: tf[egitim_kumesi_indexleri[sinif], :]
        for sinif in sinif_frekanslari
    }

    test_kumesi = {
        sinif: tf[test_kumesi_indexleri[sinif], :]
        for sinif in sinif_frekanslari
    }

    return egitim_kumesi, test_kumesi


def egitim(egitim_kumesi: dict[int, csc_matrix]) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    kelime_olasiliklari = {}

    Nc = {sinif: egitim_kumesi[sinif].shape[0] for sinif in egitim_kumesi}

    N = sum(Nc.values())

    priori_olasilik = {sinif: math.log10(Nc[sinif] / N) for sinif in egitim_kumesi}

    B = list(egitim_kumesi.values())[0].shape[1]

    STct = {
        sinif: sum(egitim_kumesi[sinif].data) + B for sinif in egitim_kumesi
    }

    for sinif in egitim_kumesi:
        tf_c = egitim_kumesi[sinif]
        Tc = np.array(tf_c.sum(axis=0)).flatten() + 1
        Ptc = Tc / STct[sinif]
        LPtc = np.log10(Ptc)

        kelime_olasiliklari[sinif] = LPtc.tolist()

    return priori_olasilik, kelime_olasiliklari


def sinifla(ornek: csr_matrix, priori_olasilik: dict[int, list[float]],
            kelime_olasilik: dict[int, list[float]]) -> int:
    olasiliklar = []
    for sinif in priori_olasilik:
        kelime_olasiliklari = np.array(kelime_olasilik[sinif])
        ornek_kelime_olasiliklari = kelime_olasiliklari[ornek.indices] * ornek.data
        log_olasilik = priori_olasilik[sinif] + sum(ornek_kelime_olasiliklari.tolist())
        olasiliklar.append((sinif, log_olasilik))
    olasiliklar.sort(key=lambda x: x[1], reverse=True)
    return olasiliklar[0][0]


def basarim_testi(test_kumesi: dict[int, csc_matrix], priori_olasilik: dict[int, list[float]],
                  kelime_olasilik: dict[int, list[float]]):
    siniflama_sonuclari = {}

    for sinif in test_kumesi:
        ornekler = test_kumesi[sinif].tocsr(copy=True)
        tahminler = [sinifla(ornekler[i], priori_olasilik, kelime_olasilik)
                     for i in range(ornekler.shape[0])]
        siniflama_sonuclari[sinif] = np.array(tahminler)
        del ornekler
    DogruSinifi = 1
    YanlisSinifi = 0

    DogruTahminEdilenDogrular = np.sum(siniflama_sonuclari[DogruSinifi] == DogruSinifi)
    DogruTahminEdilenYanlislar = np.sum(siniflama_sonuclari[DogruSinifi] != DogruSinifi)
    YanlisTahminEdilenYanlislar = np.sum(siniflama_sonuclari[YanlisSinifi] == YanlisSinifi)
    YanlisTahminEdilenDogrular = np.sum(siniflama_sonuclari[YanlisSinifi] != YanlisSinifi)
    TruePositive = DogruTahminEdilenDogrular
    TrueNegative = YanlisTahminEdilenYanlislar
    FalsePositive = DogruTahminEdilenYanlislar
    FalseNegative = YanlisTahminEdilenDogrular
    TP = TruePositive
    TN = TrueNegative
    FP = FalsePositive
    FN = FalseNegative

    PP = TP + FP
    PN = FN + TN
    P = TP + FN
    N = FP + TN

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1
    }


def siniflandirma():
    veriler = veri_hazirlama()
    tf, siniflar = sayisallastir(veriler)
    egitim_kumesi, test_kumesi = veriyi_bol(tf, siniflar, 0.66)
    prioriler, kelime_olasiliklari = egitim(egitim_kumesi)
    pprint(basarim_testi(test_kumesi, prioriler, kelime_olasiliklari))


def main():
    siniflandirma()


if __name__ == '__main__':
    main()
