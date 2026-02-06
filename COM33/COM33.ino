void setup() {
  Serial.begin(115200);
  delay(1000);                 // give serial time
  Serial.println("Arduino ready");
}

void loop() {
  Serial.println("Hello from Arduino");
  delay(1000);
}
