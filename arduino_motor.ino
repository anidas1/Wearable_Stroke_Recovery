/*
 * arduino_motor.ino — Adafruit Motor Shield 2.0
 *
 * MANUAL (serial 115200):
 *   '1' = M1–M4 ON, 'x' = OFF
 *
 * INFERENCE (ESP32 GPIO17 → D2, matches sampler_troubleshooter.ino):
 *   One *rising edge* on REACH (LOW→HIGH) starts a fixed 5 s motor burst.
 *   After 5 s, motors stop and logic x*locks* until you press 'k' to arm again.
 *   If REACH is still HIGH when you arm, you must wait for LOW before the
 *   next HIGH can trigger (so one long REACH doesn’t chain bursts).
 *
 *   'f' — enter inference + arm
 *   'k' — re-arm after a burst (required before the next REACH can trigger)
 *   'm' — manual mode
 *   '?' — help
 */

#include <Wire.h>
#include <Adafruit_MotorShield.h>

#define REACH_INPUT_PIN 2
#define BURST_MS        3000UL

Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor* motor1 = AFMS.getMotor(1);
Adafruit_DCMotor* motor2 = AFMS.getMotor(2);
Adafruit_DCMotor* motor3 = AFMS.getMotor(3);
Adafruit_DCMotor* motor4 = AFMS.getMotor(4);

#define MOTOR_SPEED 77

enum RunMode { MODE_MANUAL, MODE_INFERENCE };
RunMode runMode = MODE_MANUAL;

// Inference sub-state
enum InferPhase { INF_ARMED, INF_BURST, INF_LOCKED };
InferPhase inferPhase = INF_LOCKED;
unsigned long burstEndMs = 0;
bool blockUntilReachLow = false;
bool prevReachHigh = false;

static void allForward() {
  motor1->setSpeed(MOTOR_SPEED);
  motor2->setSpeed(MOTOR_SPEED);
  motor3->setSpeed(MOTOR_SPEED);
  motor4->setSpeed(MOTOR_SPEED);
  motor1->run(FORWARD);
  motor2->run(FORWARD);
  motor3->run(FORWARD);
  motor4->run(FORWARD);
}

static void allRelease() {
  motor1->run(RELEASE);
  motor2->run(RELEASE);
  motor3->run(RELEASE);
  motor4->run(RELEASE);
}

static void armInferenceListening() {
  inferPhase = INF_ARMED;
  bool hi = digitalRead(REACH_INPUT_PIN) == HIGH;
  blockUntilReachLow = hi;
  prevReachHigh = hi;
  if (blockUntilReachLow) {
    Serial.println(F("Armed: wait for REACH LOW, then next HIGH → 5s burst."));
  } else {
    Serial.println(F("Armed: next REACH HIGH (edge) → 5s burst."));
  }
}

static void printHelp() {
  Serial.println(F("=== arduino_motor ==="));
  Serial.println(F("'m' manual | 'f' inference | 'k' re-arm after burst"));
  Serial.println(F("Manual: '1' ON, 'x' OFF"));
  Serial.println(F("Inference: one HIGH edge → 5s ON, then press 'k' to arm again"));
  Serial.println(F("'?' help"));
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;
  }

  pinMode(REACH_INPUT_PIN, INPUT);

  if (!AFMS.begin()) {
    Serial.println(F("Motor Shield not found — check stacking and I2C."));
    while (1) {
      delay(1000);
    }
  }

  allRelease();
  Serial.println(F("Ready — MANUAL default. '?' for commands."));
  printHelp();
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    while (Serial.available()) {
      Serial.read();
    }

    if (cmd == 'm' || cmd == 'M') {
      runMode = MODE_MANUAL;
      inferPhase = INF_LOCKED;
      allRelease();
      Serial.println(F("Mode: MANUAL (1/x)"));
    } else if (cmd == 'f' || cmd == 'F') {
      runMode = MODE_INFERENCE;
      allRelease();
      Serial.println(F("Mode: INFERENCE"));
      armInferenceListening();
    } else if (cmd == 'k' || cmd == 'K') {
      if (runMode == MODE_INFERENCE && inferPhase == INF_LOCKED) {
        armInferenceListening();
      } else if (runMode == MODE_INFERENCE) {
        Serial.println(F("'k' only after 5s burst ends (currently armed or running)."));
      } else {
        Serial.println(F("Not in inference mode — press 'f' first."));
      }
    } else if (cmd == '?') {
      printHelp();
    } else if (runMode == MODE_MANUAL) {
      if (cmd == '1') {
        allForward();
        Serial.println(F("M1–M4 ON"));
      } else if (cmd == 'x' || cmd == 'X') {
        allRelease();
        Serial.println(F("M1–M4 OFF"));
      }
    }
  }

  if (runMode != MODE_INFERENCE) {
    return;
  }

  bool hi = digitalRead(REACH_INPUT_PIN) == HIGH;

  if (inferPhase == INF_ARMED) {
    if (blockUntilReachLow) {
      if (!hi) {
        blockUntilReachLow = false;
        prevReachHigh = false;
      }
    } else {
      if (hi && !prevReachHigh) {
        allForward();
        burstEndMs = millis() + BURST_MS;
        inferPhase = INF_BURST;
        Serial.println(F("REACH edge — motors 5 s"));
      }
      prevReachHigh = hi;
    }
  } else if (inferPhase == INF_BURST) {
    if ((long)(millis() - burstEndMs) >= 0) {
      allRelease();
      inferPhase = INF_LOCKED;
      Serial.println(F("5 s done — press 'k' to arm for next REACH."));
    }
  }
  // INF_LOCKED: nothing until 'k'
}
