from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

input_string = ("Korkma, sönmez bu şafaklarda yüzen al sancak; "
                "Sönmeden yurdumun üstünde tüten en son ocak. "
                "O benim milletimin yıldızıdır, parlayacak; "
                "O benimdir, o benim milletimindir ancak. "
                "Çatma, kurban olayım çehreni ey nazlı hilâl! "
                "Kahraman ırkıma bir gül… ne bu şiddet bu celâl? "
                "Sana olmaz dökülen kanlarımız sonra helâl, "
                "Hakkıdır, Hakk’a tapan, milletimin istiklâl. "
                "Ben ezelden beridir hür yaşadım, hür yaşarım. "
                "Hangi çılgın bana zincir vuracakmış? Şaşarım! "
                "Kükremiş sel gibiyim; bendimi çiğner, aşarım; "
                "Yırtarım dağları, enginlere sığmam, taşarım. "
                "Garb’ın âfâkını sarmışsa çelik zırhlı duvar; "
                "Benim iman dolu göğsüm gibi serhaddim var. "
                "Ulusun, korkma! Nasıl böyle bir îmânı boğar, "
                "Medeniyet! dediğin tek dişi kalmış canavar? "
                "Arkadaş! Yurduma alçakları uğratma sakın; "
                "Siper et gövdeni, dursun bu hayâsızca akın. "
                "Doğacaktır sana va’dettiği günler Hakk’ın… "
                "Kim bilir, belki yarın… belki yarından da yakın. "
                "Bastığın yerleri toprak! diyerek geçme, tanı! "
                "Düşün altındaki binlerce kefensiz yatanı. "
                "Sen şehîd oğlusun, incitme, yazıktır atanı; "
                "Verme, dünyâları alsan da, bu cennet vatanı. "
                "Kim bu cennet vatanın uğruna olmaz ki fedâ? "
                "Şühedâ fışkıracak, toprağı sıksan şühedâ! "
                "Cânı, cânânı, bütün varımı alsın da Hudâ, "
                "Etmesin tek vatanımdan beni dünyâda cüdâ. "
                "Ruhumun senden, İlâhî, şudur ancak emeli: "
                "Değmesin ma’bedimin göğsüne nâ-mahrem eli! "
                "Bu ezanlar-ki şehâdetleri dînin temeli- "
                "Ebedî yurdumun üstünde benim inlemeli "
                "O zaman vecd ile bin secde eder –varsa- taşım; "
                "Her cerîhamdan, İlâhî, boşanıp kanlı yaşım, "
                "Fışkırır rûh-i mücerred gibi yerden na’şım; "
                "O zaman yükselerek Arş’a değer, belki başım. "
                "Dalgalan sen de şafaklar gibi ey şanlı hilâl; "
                "Olsun artık dökülen kanlarımın hepsi helâl. "
                "Ebediyen sana yok, ırkıma yok izmihlâl: "
                "Hakkıdır, hür yaşamış bayrağımın hürriyet; "
                "Hakkıdır, Hakk’a tapan milletimin istiklâl!")

output = tokenizer.encode(input_string).ids

print(output)

pencere_boyutu = 110

for i in range(len(output) - pencere_boyutu + 1):
    window = output[i:i + pencere_boyutu]
    print(window)

kavram_boyutu = 5

for i in range(len(output)):
    oncesi = i - kavram_boyutu
    sonrasi = i + kavram_boyutu + 1

    once_vektor = []
    while oncesi < 0:
        once_vektor.append('<PADDING>')
        oncesi += 1

    once_vektor += output[oncesi:i]

    sonra_vektor = output[(i + 1):min(len(output), sonrasi)]
    sonra_vektor += ['<PADDING>'] * max(0, kavram_boyutu - len(sonra_vektor))

    print(once_vektor, output[i], sonra_vektor)
