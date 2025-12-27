import smbus2
import time

class MAX30102:
    def __init__(self, bus=1, address=0x57):
        self.bus = smbus2.SMBus(bus)
        self.address = address
        self.reset()
        self.setup()

    def write_reg(self, reg, val):
        self.bus.write_byte_data(self.address, reg, val)

    def read_reg(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def reset(self):
        # Işığı söndüren kilitlenme burada çözülüyor
        self.write_reg(0x09, 0x40)
        time.sleep(0.6) # Reset sonrası sensörün kendine gelmesi için şart

    def setup(self):
        # Işığı yakan ve açık tutan ayarlar
        self.write_reg(0x09, 0x03) # SpO2 modu (Hem Kırmızı hem IR LED aktif)
        time.sleep(0.1)
        self.write_reg(0x08, 0x4f) # FIFO konfigürasyonu
        self.write_reg(0x0a, 0x27) # 100Hz örnekleme
        self.write_reg(0x0c, 0x3F) # RED LED gücü (Maksimuma yakın)
        self.write_reg(0x0d, 0x3F) # IR LED gücü
        print("💡 Donanım Ayarları Yüklendi, Işık Yakıldı.")

    def read_fifo(self):
        try:
            # FIFO register'ından 6 byte veri oku
            d = self.bus.read_i2c_block_data(self.address, 0x07, 6)
            red = (d[0] << 16 | d[1] << 8 | d[2]) & 0x03FFFF
            ir = (d[3] << 16 | d[4] << 8 | d[5]) & 0x03FFFF
            return red, ir
        except Exception as e:
            return 0, 0
