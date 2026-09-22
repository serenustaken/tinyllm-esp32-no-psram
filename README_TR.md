# Klasik ESP32'de TinyLLM — PSRAM Yok

Bu projede, 260 bin parametreli TinyStories dil modelini harici PSRAM bulunmayan
klasik ESP32/ESP-32S kartta tamamen çevrimdışı çalıştırıyorum.

Bu bir ChatGPT alternatifi değildir. Kısa İngilizce çocuk hikâyeleri üretmek
için eğitilmiş, çok küçük ve deneysel bir modeldir. Wi-Fi, bulut servisi, API,
SD kart veya harici sunucu kullanmaz.

## Neden paylaşmaya değer?

Benzer ESP32 dil modeli projeleri çoğunlukla daha yeni ESP32-S3 ve harici PSRAM
kullanıyor. Bu sürümde model flash bellekte tutuluyor, Q8_0 biçiminde
küçültülüyor ve 64 tokenlık sınırlı bir çalışma belleği kullanılıyor. Böylece
yaygın klasik ESP32 kartta Arduino IDE üzerinden çalışabiliyor.

Bu projeye pratik bir soruyla başladım: Gerçek bir dil modeli, yaygın olarak
bulunan ve PSRAM içermeyen klasik bir ESP32 üzerinde yerel olarak çalışabilir
mi? Mevcut açık kaynak çalışmalardan yararlanarak bellek kullanımına dikkat
eden, test edilmiş, belgelenmiş ve başkalarının da uygulayabileceği bir Arduino
çözümü hazırladım.

Araştırma, uygulama, hata ayıklama ve belgelendirme süreçlerinin bazı
bölümlerinde yapay zekâ araçlarından yararlandım. Sistemin tamamını derledim,
fiziksel karta yükledim ve gerçek donanım üzerinde doğruladım. Bu projeyle
pratik entegrasyona, tekrarlanabilirliğe ve gömülü yapay zekâyı daha erişilebilir
hâle getirmeye odaklanıyorum.

Projeyi [@serenustaken](https://github.com/serenustaken) olarak sürdürüyorum.

## Doğrulanan değerler

| Özellik | Değer |
|---|---:|
| Kart | Klasik ESP32 / ESP-32S |
| Harici PSRAM | Gerekmiyor |
| Model | TinyStories 260K |
| Model biçimi | llama2.c v2 Q8_0 |
| Quantization group size | 4 |
| Model boyutu | 521.728 bayt |
| Tokenizer | 6.227 bayt |
| Bağlam | 64 token |
| Çalışma tamponları | Yaklaşık 89.400 bayt |
| Derlenmiş program | 829.828 bayt |
| Statik RAM | 22.996 bayt |

Arduino çekirdeği sürümüne göre derleme değerleri küçük farklılık gösterebilir.

## Kurulum

1. Bu depoyu indir veya klonla.
2. Arduino IDE'de
   `tinyllm_esp32_no_psram/tinyllm_esp32_no_psram.ino` dosyasını aç.
3. **Araçlar** menüsünde şu ayarları seç:

   | Ayar | Değer |
   |---|---|
   | Board | ESP32 Dev Module |
   | CPU Frequency | 240MHz (WiFi/BT) |
   | Flash Frequency | 80MHz |
   | Flash Mode | QIO |
   | Flash Size | 4MB (32Mb) |
   | Partition Scheme | Huge APP (3MB No OTA/1MB SPIFFS) |
   | PSRAM | Disabled |
   | Upload Speed | Önce 921600; sorun olursa 115200 |

4. Doğru USB portunu seçip **Upload** düğmesine bas.
5. Serial Monitor'ü aç, hızı `115200 baud`, satır sonunu `New Line` yap.
6. İlk otomatik hikâyeden sonra kısa bir İngilizce başlangıç cümlesi yaz.

Örnekler:

```text
Once there was a little dog
Lily went to the forest
Tom found a red ball
```

## Bilmen gereken sınırlar

- Model yalnızca kısa İngilizce hikâyelerde anlamlı sonuç vermeye çalışır.
- Soru-cevap modeli değildir ve verdiği bilgiye güvenilmemelidir.
- Girdi ve üretilen metin toplamda 64 tokenlık bağlama sığmalıdır.
- Klasik ESP32 üzerinde üretim bilgisayara göre yavaştır.
- Yeni bir cümle göndermeden önce mevcut üretimin bitmesini beklemelisin.

## Sık karşılaşılan sorunlar

- **`llm_engine.h` bulunamıyor:** Yalnızca `.ino` dosyasını taşımayın; bütün
  sketch klasörünü birlikte açın.
- **`Connecting...` ekranında kalıyor:** Yazma başlayana kadar `BOOT` tuşunu
  basılı tutun.
- **`resource busy`:** Portu kullanan diğer Serial Monitor veya terminali
  kapatın. Aynı bağlantıyı iki program kullanamaz.
- **Yetersiz RAM:** Wi-Fi/Bluetooth kodu eklemeyin veya `CONTEXT_TOKENS`
  değerini 48/32 yapın.
- **Anlamsız çıktı:** Açılışta model boyutunun `521728` olduğunu kontrol edin.

Daha ayrıntılı fakat sade teknik açıklama için
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) dosyasını okuyabilirsin.

## Kaynaklar ve dürüst atıf

Bu proje başta [llama2.c](https://github.com/karpathy/llama2.c) olmak üzere açık
kaynak çalışmalara dayanır. DaveBben ve doryiii'nin ESP32-S3 projeleri de fikir
ve uygulama referansı sağlamıştır. Ayrıntılar [NOTICE.md](NOTICE.md) içindedir.

Kod MIT lisansı altında paylaşılır. Modelin yayınlandığı `tinyllamas` deposu da
MIT lisanslı olarak işaretlenmiştir.
