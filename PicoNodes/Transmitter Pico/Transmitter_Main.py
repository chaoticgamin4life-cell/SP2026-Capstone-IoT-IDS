from sx1262 import SX1262
import time
import urandom

# Initialize LoRa
sx = SX1262(spi_bus=1, clk=10, mosi=11, miso=12, cs=3, irq=20, rst=15, gpio=2)

sx.begin(freq=923, bw=500.0, sf=12, cr=8, syncWord=0x12,
         power=-5, currentLimit=60.0, preambleLength=8,
         implicit=False, implicitLen=0xFF,
         crcOn=True, txIq=False, rxIq=False,
         tcxoVoltage=1.7, useRegulatorLDO=False, blocking=True)

NODE_ID = "Pico_Node_01"

malicious_burst = 0  # how many malicious packets remain in this burst
seq = 0
while True:
    
    seq += 1
    timestamp = seq

    # If not in a malicious burst, decide whether to start one
    if malicious_burst == 0:
        start_attack = (urandom.getrandbits(8) / 255.0) < 0.25  # 25% chance

        if start_attack:
            malicious_burst = urandom.getrandbits(3) + 5
            if malicious_burst > 10:
                malicious_burst = 10

    # Determine if this packet is malicious
    is_malicious = 1 if malicious_burst > 0 else 0

    if is_malicious:
        # Malicious behaviour — manipulated timestamp, fast interval
        offset   = urandom.getrandbits(10) - 512
        interval = 0.5 + (urandom.getrandbits(4) / 10.0)
        malicious_burst -= 1

    else:
        # Normal behaviour — small jitter, 10s interval
        offset   = urandom.getrandbits(6) - 32
        interval = 10

    timestamp = timestamp + offset

    payload = "ID={},TS={},M={}".format(
        NODE_ID, timestamp, is_malicious
    ).encode()

    print("Sending (M={}): {}".format(is_malicious, payload))
    sx.send(payload)

    time.sleep(interval)

