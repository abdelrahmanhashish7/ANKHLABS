#include <Wire.h>
#include "MAX30105.h"
#include <LiquidCrystal_I2C.h>
#include <math.h>

#define SAMPLE_RATE 100
#define BUFFER_SIZE 200
#define FEATURE_INTERVAL 5000
#define IR_THRESHOLD 80000

MAX30105 sensor;
LiquidCrystal_I2C lcd(0x27, 16, 2);

enum State { NO_FINGER, BASELINE, PREDICT };
State state = NO_FINGER;

float irBuf[BUFFER_SIZE];
float redBuf[BUFFER_SIZE];
unsigned long tBuf[BUFFER_SIZE];
int idx = 0;

float ratio_base = 0, ac_base = 0, dc_base = 0, pi_base = 0, slope_base = 0;
int baselineCount = 0;
#define BASELINE_WINDOWS 6   // 6 × 5s = 30 sec

unsigned long lastFeatureTime = 0;

#define RATIO_M  -0.000522
#define RATIO_S   0.033974
#define AC_M     15.122963
#define AC_S    412.982216
#define DC_M   -423.108476
#define DC_S  12556.207636
#define PI_M     0.000133
#define PI_S     0.003568
#define SLOPE_M -0.009106
#define SLOPE_S  0.112513

#define B0 117.746339
#define B1 -1.468293
#define B2 -18.740200
#define B3 -0.025717
#define B4 18.321776
#define B5 0.614418

#define GLUCOSE_OFFSET -11.5

void setup() {
  Serial.begin(115200);
  Wire.begin();

  lcd.init();
  lcd.backlight();
  lcd.print("Glucose Est.");

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    lcd.clear();
    lcd.print("Sensor Error");
    while (1);
  }

  sensor.setup(0x1F, 4, 2, SAMPLE_RATE, 411, 4096);
  delay(1500);
  lcd.clear();
  lcd.print("Place Finger");
}

void loop() {
  unsigned long now = millis();

  if (idx < BUFFER_SIZE) {
    irBuf[idx] = sensor.getIR();
    redBuf[idx] = sensor.getRed();
    tBuf[idx] = now;
    idx++;
  }

  float irMean = 0;
  for (int i = 0; i < idx; i++) irMean += irBuf[i];
  irMean /= max(idx, 1);

  if (irMean < IR_THRESHOLD) {
    state = NO_FINGER;
    idx = 0;
    lcd.clear();
    lcd.print("No Finger");
    delay(300);
    return;
  }

  if (state == NO_FINGER && irMean > IR_THRESHOLD) {
    state = BASELINE;
    baselineCount = 0;
    ratio_base = ac_base = dc_base = pi_base = slope_base = 0;
    lcd.clear();
    lcd.print("Calibrating...");
    idx = 0;
    return;
  }

  if (now - lastFeatureTime < FEATURE_INTERVAL || idx < 30) return;
  lastFeatureTime = now;

  float irM = 0, redM = 0;
  for (int i = 0; i < idx; i++) {
    irM += irBuf[i];
    redM += redBuf[i];
  }
  irM /= idx;
  redM /= idx;

  float ratio = irM / (redM + 1e-6);
  float acSum = 0;
  for (int i = 0; i < idx; i++) {
    float x = irBuf[i] - irM;
    acSum += x * x;
  }

  float AC = sqrt(acSum / idx);
  float DC = irM;
  float PI_feature = AC / (DC + 1e-6);
  Serial.print("PI Feature: ");
  Serial.println(PI_feature, 6);

  float sumT = 0, sumY = 0, sumTT = 0, sumTY = 0;
  for (int i = 0; i < idx; i++) {
    sumT += tBuf[i];
    sumY += irBuf[i];
    sumTT += tBuf[i] * tBuf[i];
    sumTY += tBuf[i] * irBuf[i];
  }

  float slope = (idx * sumTY - sumT * sumY) /
                (idx * sumTT - sumT * sumT + 1e-6);
  idx = 0;

  if (state == BASELINE) {
    ratio_base += ratio;
    ac_base += AC;
    dc_base += DC;
    pi_base += PI_feature;
    slope_base += slope;
    baselineCount++;

    if (baselineCount >= BASELINE_WINDOWS) {
      ratio_base /= baselineCount;
      ac_base /= baselineCount;
      dc_base /= baselineCount;
      pi_base /= baselineCount;
      slope_base /= baselineCount;

      state = PREDICT;
      lcd.clear();
      lcd.print("Measuring...");
    }
    return;
  }

  float ratio_n = ((ratio - ratio_base) - RATIO_M) / RATIO_S;
  float ac_n    = ((AC - ac_base) - AC_M) / AC_S;
  float dc_n    = ((DC - dc_base) - DC_M) / DC_S;
  float pi_n    = ((PI_feature - pi_base) - PI_M) / PI_S;
  float slope_n = ((slope - slope_base) - SLOPE_M) / SLOPE_S;

  float glucose =
    B0 +
    B1 * ratio_n +
    B2 * ac_n +
    B3 * dc_n +
    B4 * pi_n +
    B5 * slope_n;

  glucose += GLUCOSE_OFFSET;

  if (isnan(glucose) || glucose < 40 || glucose > 400) return;

  Serial.print("Glucose Est: ");
  Serial.println(glucose, 1);

  lcd.clear();
  lcd.print("PI:");
  lcd.print(PI_feature, 4);
  lcd.setCursor(0, 1);
  lcd.print("G:");
  lcd.print(glucose, 1);
}
