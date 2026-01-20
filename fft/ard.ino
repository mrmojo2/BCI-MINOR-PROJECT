void setup() {
  Serial.begin(115200);   // Start USB serial communication
}

void loop() {
  int adcValue = analogRead(A0);  // Read analog input (0–1023)
  Serial.println(adcValue);       // Send value over USB
  delay(1);                       // Small delay for stability
}

