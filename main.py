# Wikipedi dosyasındaki text bilgisine erişilmeli!

# DENEME 2
# JSON görünüyor, Python JSON kütüphanesi ile bilgi almayı deneyelim

from json import loads

# loads bir metin bilgisini Python nesnesine dönüştürür

orneklem = []

with open("veri/orneklem.json", "r") as dosya:
    for satir in dosya:
        ornek = loads(satir)
        orneklem.append(ornek)

print(f"Örneklem sayısı {len(orneklem)}")
