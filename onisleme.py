import re
import unicodedata

import nltk

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
