#include <Wire.h>
#include <Adafruit_MotorShield.h>

Adafruit_MotorShield AFMS = Adafruit_MotorShield();
Adafruit_DCMotor* motor3  = AFMS.getMotor(3);

#define MOTOR_SPEED  77
#define SIGNAL_PIN   2

void setup() {
  Serial.begin(115200);
  pinMode(SIGNAL_PIN, INPUT);
  AFMS.begin();
  motor3->setSpeed(0);
  motor3->run(RELEASE);
  Serial.println("Ready");
}

void loop() {
  if (digitalRead(SIGNAL_PIN) == HIGH) {
    motor3->setSpeed(MOTOR_SPEED);
    motor3->run(FORWARD);
    Serial.println("REACH - Motor ON");
  } else {
    motor3->run(RELEASE);
  }

  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '1') { motor3->setSpeed(MOTOR_SPEED); motor3->run(FORWARD); Serial.println("Manual ON"); }
    else if (cmd == 'x') { motor3->run(RELEASE); Serial.println("Manual OFF"); }
  }

  delay(10);
}
