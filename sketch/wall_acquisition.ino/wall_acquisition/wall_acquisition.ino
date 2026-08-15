// ThermalGuard Edge — per-wall acquisition node
// Classic Arduino Uno, one 1-Wire bus on D2 (all DS18B20 DATA lines,
// single 4.7k pull-up D2->5V; VDD->5V, GND->GND common rails).
// Streams one JSON frame per cycle over USB serial @115200 — consumed
// by frame_server.py on the acquisition host. No String buffering
// (2KB RAM discipline): fields print directly.

#include <OneWire.h>
#include <DallasTemperature.h>

#define BUS_PIN 2
#define MAX_SENSORS 100
#define CYCLE_MS 2000

OneWire wire(BUS_PIN);
DallasTemperature bus(&wire);
DeviceAddress addrs[MAX_SENSORS];
uint8_t nSensors = 0;
uint32_t seq = 0;

void printAddr(const DeviceAddress a) {
  for (uint8_t i = 0; i < 8; i++) {
    if (a[i] < 16) Serial.print('0');
    Serial.print(a[i], HEX);
  }
}

void setup() {
  Serial.begin(115200);
  bus.begin();
  nSensors = bus.getDeviceCount();
  if (nSensors > MAX_SENSORS) nSensors = MAX_SENSORS;
  for (uint8_t s = 0; s < nSensors; s++) bus.getAddress(addrs[s], s);
  bus.setResolution(12);
  bus.setWaitForConversion(false);

  Serial.print(F("{\"census\":"));
  Serial.print(nSensors);
  Serial.println(F("}"));
}

void loop() {
  uint32_t t0 = millis();
  bus.requestTemperatures();          // broadcast convert, all in parallel
  delay(800);                         // 12-bit worst case + margin

  Serial.print(F("{\"seq\":"));
  Serial.print(seq++);
  Serial.print(F(",\"buses\":[{\"bus\":0,\"enabled\":true,\"sensors\":["));
  for (uint8_t s = 0; s < nSensors; s++) {
    if (s) Serial.print(',');
    float t = bus.getTempC(addrs[s]);
    bool ok = (t > -55.0 && t < 125.0 && t != 85.0);
    Serial.print(F("{\"rom\":\""));
    printAddr(addrs[s]);
    Serial.print(F("\",\"t\":"));
    Serial.print(t, 2);
    Serial.print(F(",\"ok\":"));
    Serial.print(ok ? F("true") : F("false"));
    Serial.print('}');
  }
  Serial.println(F("]}]}"));

  uint32_t el = millis() - t0;
  if (el < CYCLE_MS) delay(CYCLE_MS - el);
}
