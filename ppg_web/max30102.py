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
        self.write_reg(0x09, 0x40)
        time.sleep(0.6)

    def setup(self):
        self.write_reg(0x09, 0x03)
        time.sleep(0.1)
        self.write_reg(0x08, 0x4F)
        self.write_reg(0x0A, 0x27)
        self.write_reg(0x0C, 0x3F)
        self.write_reg(0x0D, 0x3F)

    def read_fifo(self):
        d = self.bus.read_i2c_block_data(self.address, 0x07, 6)
        red = (d[0] << 16 | d[1] << 8 | d[2]) & 0x03FFFF
        ir = (d[3] << 16 | d[4] << 8 | d[5]) & 0x03FFFF
        return red, ir
