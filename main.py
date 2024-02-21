# Wikipedi dosyasındaki text bilgisine erişilmeli!

# DENEME 1 (BAŞARISIZ)
# JSON görünüyor, Python JSON kütüphanesi ile bilgi almayı deneyelim

from json import load
# load bir json dosyasını Python nesnesine dönüştürür

with open("veri/orneklem.json", "r") as dosya:
    orneklem = load(dosya)

print(f"Örneklem sayısı {len(orneklem)}")
