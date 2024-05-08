import os
from pathlib import Path

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import NFKC, StripAccents, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

if not os.path.exists("tokenizer.json"):
    kaynak_klasor = Path('veri/42bin_haber')

    text_dosyalari = list([str(dosya) for dosya in kaynak_klasor.glob("**/*.txt")])

    unk_token = "<BİLİNMİYOR>"
    spl_tokens = ["<BİLİNMİYOR>", "<AYIRAÇ>", "<MASKE>", "<SINIF AYIRACI>"]  # special tokens

    tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
    trainer = WordPieceTrainer(vocab_size=32 * 1024, special_tokens=spl_tokens)

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = normalizers.Sequence([NFKC(), StripAccents()])

    tokenizer.train(text_dosyalari, trainer=trainer)
else:
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

output = tokenizer.encode(input_string).tokens

print(output)

# tokenizer.save("tokenizer.json")
