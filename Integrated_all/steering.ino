#include <Car_Library.h>

// --- 모터 핀 설정 ---
int motorC1 = 2; // Steering Motor IN1
int motorC2 = 3; // Steering Motor IN2
int motorA1 = 4; // Drive Motor A IN1
int motorA2 = 5; // Drive Motor A IN2
int motorB1 = 7; // Drive Motor B IN1
int motorB2 = 6; // Drive Motor B IN2 

// --- 설정 변수 ---
int steerSpeed = 200;
int driveSpeed = 0;   
int analogPin = A0;   
bool isDriveFwd = true; // 전진 모드 플래그

// --- 로직 상태 ---
#define STATE_STOP 0
#define STATE_FWD 1   // (조향 모터 기준)
#define STATE_BWD 2   // (조향 모터 기준)

int target = 60;      // 중앙값
int deadzone = 2;
bool isAuto = false;  // 시리얼 자동 모드 여부 (현재는 's' 수신 시 true로만 바뀜)
int desiredState = STATE_STOP;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10);

  pinMode(motorA1, OUTPUT); pinMode(motorA2, OUTPUT);
  pinMode(motorB1, OUTPUT); pinMode(motorB2, OUTPUT);
  pinMode(motorC1, OUTPUT); pinMode(motorC2, OUTPUT);

  motor_hold(motorC1, motorC2);
  motor_hold(motorA1, motorA2);
  motor_hold(motorB1, motorB2);
}

void loop() {
  // 1. 가변저항(조향 위치 피드백) 읽기
  int val = potentiometer_Read(analogPin);

  // 2. 시리얼(Python) 입력 처리
  if (Serial.available()) {
    char c = Serial.read();

    // 조향 목표값 설정: "s<각도값>\n" 예) s60
    if (c == 's') {
      int inputVal = Serial.parseInt();
      if (inputVal >= 41 && inputVal <= 79) {
        isAuto = true;
        target = inputVal;
      }
    }
    // 속도 설정: "v<속도>\n" 예) v120, 후진은 음수 예) v-80
    else if (c == 'v') {
      int v = Serial.parseInt();
      if (v < 0) {
        isDriveFwd = false;
        driveSpeed = abs(v);
      } else {
        isDriveFwd = true;
        driveSpeed = v;
      }
      driveSpeed = constrain(driveSpeed, 0, 255);
    }
  }

  // 3. 조향 모터 제어 (Motor C)
  int error = target - val;
  if (abs(error) <= deadzone) desiredState = STATE_STOP;
  else if (val < target) desiredState = STATE_FWD;
  else desiredState = STATE_BWD;

  if (desiredState == STATE_STOP) motor_hold(motorC1, motorC2);
  else if (desiredState == STATE_FWD) motor_backward(motorC1, motorC2, steerSpeed);
  else if (desiredState == STATE_BWD) motor_forward(motorC1, motorC2, steerSpeed);

  // 4. 주행 모터 제어 (Motor A, B)
  if (driveSpeed > 0) {
    if (isDriveFwd) {
      motor_forward(motorA1, motorA2, driveSpeed);
      motor_forward(motorB1, motorB2, driveSpeed);
    } else {
      motor_backward(motorA1, motorA2, driveSpeed);
      motor_backward(motorB1, motorB2, driveSpeed);
    }
  } else {
    motor_hold(motorA1, motorA2);
    motor_hold(motorB1, motorB2);
  }

  delay(10);
}
