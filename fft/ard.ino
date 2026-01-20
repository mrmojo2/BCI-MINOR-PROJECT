const uint32_t FS = 1000;
const uint32_t TS_US = 1000000UL / FS;

void setup() {
  Serial.begin(230400); // faster helps
}

void loop() {
  static uint32_t next = micros();
  if ((int32_t)(micros() - next) >= 0) {
    next += TS_US;
    uint16_t adc = analogRead(A0);
    // Send as 2 bytes (little endian)
    Serial.write((uint8_t*)&adc, 2);
  }
}

