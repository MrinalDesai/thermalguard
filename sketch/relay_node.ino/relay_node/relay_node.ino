// ThermalGuard Edge — relay node (isolation actuator)
// ACTIVE-LOW relay module on D7 (VCC->5V, GND->GND, IN->D7, VR bridged to VCC).
// Host sends one character over USB serial @115200:
//   'I' = engage relay -> isolation contact opens the load path
//   'C' = release      -> load restored
// 3 audible proof-clicks at boot confirm wiring.

#define RELAY_PIN 7

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);          // HIGH = released (active-LOW)
  Serial.println("relay node (active-LOW): 3 boot clicks...");
  for (int i = 0; i < 3; i++) {
    digitalWrite(RELAY_PIN, LOW);  delay(400);   // engage
    digitalWrite(RELAY_PIN, HIGH); delay(400);   // release
  }
  Serial.println("ready. I = engage (isolate), C = release");
}

void loop() {
  if (!Serial.available()) return;
  char c = Serial.read();
  if (c == 'I' || c == 'i') { digitalWrite(RELAY_PIN, LOW);  Serial.println("RELAY: ENGAGED (isolated)"); }
  if (c == 'C' || c == 'c') { digitalWrite(RELAY_PIN, HIGH); Serial.println("RELAY: released (closed)"); }
}
