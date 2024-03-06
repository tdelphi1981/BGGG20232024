# Wikipedi dosyasındaki text bilgisine erişilmeli!

# DENEME 2
# JSON görünüyor, Python JSON kütüphanesi ile bilgi almayı deneyelim

from json import loads

# loads bir metin bilgisini Python nesnesine dönüştürür
print("1- Veri Yükleme")
orneklem = []

with open("veri/orneklem.json", "r") as dosya:
    for satir in dosya:
        ornek = loads(satir)
        orneklem.append(ornek)

print(f"Örneklem sayısı {len(orneklem)}")

# Doküman Tanımı
orneklem = [ornek["text"] for ornek in orneklem]

# Metin Önişleme


print("2- Ön işleme")

print("2.1- Karakter Önişleme")
# Karakter dönüşümü
# Python str-> lower(): küçük harfe çevirir, upper(): büyük harfe çevirir

# for i, metin in enumerate(orneklem):
#     orneklem[i] = metin.lower()
orneklem = [metin.lower() for metin in orneklem]

import unicodedata

# Unicode Normalizasyonu
# ö: 1-ö, 2-o ..
orneklem = [unicodedata.normalize('NFKD', metin) for metin in orneklem]

# Karakter Eliminasyonu
# Seçilen Karakter Kategorileri: Ll, Nd, Zs

secilen_kategoriler = ['Ll', 'Nd', 'Zs']

for i, ornek in enumerate(orneklem):
    kategoriler = [unicodedata.category(karakter) for karakter in ornek]
    yeni_metin = []
    for j, kategori in enumerate(kategoriler):
        if kategori in secilen_kategoriler:
            if kategori == 'Zs':
                yeni_metin.append(' ')
            else:
                yeni_metin.append(ornek[j])
        else:
            yeni_metin.append(' ')
    yeni_metin = "".join(yeni_metin)  # listeyi metne geri çevir
    # Yeni problem: Çoklu boşluk problemi
    while True:
        uzunluk = len(yeni_metin)
        yeni_metin = yeni_metin.replace("  ", " ")
        if len(yeni_metin) == uzunluk:
            break
    # En iyi alternatif REGEX
    orneklem[i] = yeni_metin

print("2.2- Metin Parçalama")

# for i, ornek in enumerate(orneklem):
#    orneklem[i] = ornek.split(' ')
orneklem = [ornek.split(' ') for ornek in orneklem]

print("2.3- Metin Parçası Önişleme")

import nltk

budayici = nltk.stem.SnowballStemmer('english')

zamirler = nltk.corpus.stopwords.words('english')

print(zamirler)

# for ornek in orneklem:
#    zamirsiz_ornek = [parca for parca in ornek if parca not in zamirler]
#    budanmis_ornek = [budayici.stem(parca) for parca in zamirsiz_ornek]

# for i,ornek in enumerate(orneklem):
#    orneklem[i] = [budayici.stem(parca) for parca in ornek if parca not in zamirler]

orneklem = [[budayici.stem(parca) for parca in ornek if parca not in zamirler] for ornek in orneklem]

print("3- Metin Analizi")

print("3.1- Sözlük Oluşturma")
sozluk = set()
for ornek in orneklem:
    # for kelime in ornek:
    #     if kelime not in sozluk:
    #         sozluk.append(kelime)
    sozluk.update(ornek)
    print("Sozluk boyutu: ", len(sozluk))

sozluk = list(sozluk)

sozluk.sort()

print("3.2- Ağırlıklandırma")
