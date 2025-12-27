import time


def main():
    print("== MAX30102 STREAM TEST ==")
    print("Bu test 10 saniye boyunca FIFO'dan veri okur.")

    try:
        from max30102 import MAX30102
    except Exception as e:
        print("❌ max30102 import edilemedi. test klasöründe max30102.py var mı?")
        print(f"Hata: {e}")
        return 1

    try:
        sensor = MAX30102()
    except Exception as e:
        print("❌ Sensör initialize edilemedi (I2C/izin/kablo?).")
        print(f"Hata: {e}")
        return 2

    ok = 0
    none_cnt = 0
    start = time.time()
    last_print = 0.0

    while time.time() - start < 10:
        try:
            red, ir = sensor.read_fifo()
        except Exception as e:
            print("❌ read_fifo hata verdi:")
            print(e)
            return 3

        if red is None or ir is None:
            none_cnt += 1
        else:
            ok += 1

        # saniyede ~5 satır yazdır
        now = time.time()
        if now - last_print > 0.2:
            last_print = now
            print(f"t={now-start:5.2f}s  red={red}  ir={ir}  (ok={ok}, none={none_cnt})")

        time.sleep(0.01)

    print("\n== ÖZET ==")
    print(f"Okunan geçerli örnek sayısı: {ok}")
    print(f"None dönen okuma sayısı: {none_cnt}")

    if ok > 0:
        print("✅ Veri akıyor görünüyor.")
        return 0

    print("⚠️ Hiç geçerli veri gelmedi. FIFO/konfig/bağlantı kontrol edilmeli.")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
