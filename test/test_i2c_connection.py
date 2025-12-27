import sys


def main():
    # MAX30102 default I2C address is usually 0x57
    addr = 0x57
    bus_num = 1

    print("== I2C BASIC CONNECTION TEST ==")
    print(f"Bus: /dev/i2c-{bus_num}")
    print(f"Target address: 0x{addr:02X}")

    try:
        import smbus2
    except Exception as e:
        print("❌ smbus2 import edilemedi. (pip install smbus2)")
        print(f"Hata: {e}")
        sys.exit(1)

    try:
        bus = smbus2.SMBus(bus_num)
    except Exception as e:
        print("❌ I2C bus açılamadı. I2C açık mı? (raspi-config) / izinler?")
        print(f"Hata: {e}")
        sys.exit(1)

    # MAX30102 PART ID register is 0xFF (expected 0x15)
    part_id_reg = 0xFF
    expected_part_id = 0x15

    try:
        part_id = bus.read_byte_data(addr, part_id_reg)
        print(f"✅ Cihaz cevap verdi. PART_ID (0xFF) = 0x{part_id:02X}")
        if part_id == expected_part_id:
            print("✅ PART_ID beklenen değer (0x15). MAX30102 doğru görünüyor.")
        else:
            print("⚠️ PART_ID beklenen değer değil. Adres yanlış olabilir veya farklı sensör olabilir.")
            print(f"Beklenen: 0x{expected_part_id:02X}")

    except OSError as e:
        print("❌ I2C okuma başarısız. Cihaz görünmüyor veya kablo/adres sorunu var.")
        print(f"Hata: {e}")
        sys.exit(2)
    finally:
        try:
            bus.close()
        except Exception:
            pass

    print("\nOK")


if __name__ == "__main__":
    main()
