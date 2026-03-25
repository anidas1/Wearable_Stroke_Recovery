/*
  ESP32-S3 MyoWare 3-Sensor, 9-Channel Stream

  Pinout (ADC1-only, avoids strapping pin GPIO3):
  Sensor 1: ENV=GPIO1,  RECT=GPIO2,  RAW=GPIO4
  Sensor 2: ENV=GPIO5,  RECT=GPIO6,  RAW=GPIO7
  Sensor 3: ENV=GPIO8,  RECT=GPIO9,  RAW=GPIO10

  Serial commands:
  - Signal type: 'e' = ENV, 'r' = RAW, 'c' = RECT
  - Sensor select: '1', '2', '3'
  - Multi-sensor:  'a' = all ENV, 'q' = all RECT, 'w' = all RAW
*/

#include <Arduino.h>

// ── Pin map ──────────────────────────────────────────────
const int PINS[3][3] = {
  // ENV  RECT  RAW
  {  1,    2,    4  },   // Sensor 1
  {  5,    6,    7  },   // Sensor 2
  {  8,    9,   10  }    // Sensor 3
};

// ── State ────────────────────────────────────────────────
int  activeSensor = 0;     // 0-2
int  activeSignal = 0;     // 0=ENV, 1=RECT, 2=RAW
bool allSensors   = false; // stream all 3 sensors with same signal type

const char* signalNames[] = {"ENV", "RECT", "RAW"};

// ── Helpers ──────────────────────────────────────────────
void printConfig() {
  if (allSensors) {
    Serial.printf("[Config] ALL sensors | Signal: %s\n", signalNames[activeSignal]);
  } else {
    Serial.printf("[Config] Sensor %d | Signal: %s | GPIO %d\n",
                  activeSensor + 1,
                  signalNames[activeSignal],
                  PINS[activeSensor][activeSignal]);
  }
}

void handleSerial() {
  if (!Serial.available()) return;

  char cmd = Serial.read();

  switch (cmd) {
    // Signal type
    case 'e':
      activeSignal = 0;
      allSensors = false;
      printConfig();
      break;

    case 'c':
      activeSignal = 1;
      allSensors = false;
      printConfig();
      break;

    case 'r':
      activeSignal = 2;
      allSensors = false;
      printConfig();
      break;

    // Single sensor select
    case '1':
      activeSensor = 0;
      allSensors = false;
      printConfig();
      break;

    case '2':
      activeSensor = 1;
      allSensors = false;
      printConfig();
      break;

    case '3':
      activeSensor = 2;
      allSensors = false;
      printConfig();
      break;

    // All sensors, same signal type
    case 'a':
      activeSignal = 0;   // ENV
      allSensors = true;
      printConfig();
      break;

    case 'q':
      activeSignal = 1;   // RECT
      allSensors = true;
      printConfig();
      break;

    case 'w':
      activeSignal = 2;   // RAW
      allSensors = true;
      printConfig();
      break;
  }
}

// ── Setup ────────────────────────────────────────────────
void setup() {
  Serial.begin(921600);
  delay(500);

  analogReadResolution(12);  // 0..4095

  // Configure all 9 analog inputs
  for (int s = 0; s < 3; s++) {
    for (int sig = 0; sig < 3; sig++) {
      pinMode(PINS[s][sig], INPUT);
      analogSetPinAttenuation(PINS[s][sig], ADC_11db);
    }
  }

  Serial.println("=== ESP32-S3 MyoWare 9-Channel Stream ===");
  Serial.println("Pinout:");
  Serial.println("  Sensor 1: ENV=GPIO1  RECT=GPIO2  RAW=GPIO4");
  Serial.println("  Sensor 2: ENV=GPIO5  RECT=GPIO6  RAW=GPIO7");
  Serial.println("  Sensor 3: ENV=GPIO8  RECT=GPIO9  RAW=GPIO10");
  Serial.println("Commands:");
  Serial.println("  e=ENV  c=RECT  r=RAW");
  Serial.println("  1 2 3 = select sensor");
  Serial.println("  a=all ENV  q=all RECT  w=all RAW");
  Serial.println("==========================================");
  printConfig();
}

// ── Loop ─────────────────────────────────────────────────
void loop() {
  handleSerial();

  if (allSensors) {
    int v1 = analogRead(PINS[0][activeSignal]);
    int v2 = analogRead(PINS[1][activeSignal]);
    int v3 = analogRead(PINS[2][activeSignal]);

    Serial.printf("S1_%s:%d S2_%s:%d S3_%s:%d\n",
                  signalNames[activeSignal], v1,
                  signalNames[activeSignal], v2,
                  signalNames[activeSignal], v3);
  } else {
    int val = analogRead(PINS[activeSensor][activeSignal]);
    Serial.println(val);
  }

  delay(10);  // ~100 Hz for visual inspection
