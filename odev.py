from pathlib import Path

veri_klasoru = Path(__file__).parent / "veri" / "42bin_haber" / "news" / "spor"

for dosya_yolu in veri_klasoru.glob("**/*.txt"):
    with dosya_yolu.open("r") as dosya:
        pass
