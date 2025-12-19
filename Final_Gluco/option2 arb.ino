#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

const float K_IR = -0.00204f, K_RED = -0.00036f, BIAS = 171.32f;
const int SAMPLE_COUNT = 25;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  delay(2000);
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    while (1);
  }
  particleSensor.setup(0x2F, 4, 2, 400, 411, 4096);
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeIR(0x1F);
}
void loop() {
  long irSum = 0, redSum = 0;
  int valid = 0;
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    long ir = particleSensor.getIR();
    long red = particleSensor.getRed();
    if (ir > 10000) {
      irSum += ir;
      redSum += red;
      valid++;
    }
    
    particleSensor.nextSample();
    delay(400);
  }

  if (valid > 15) {
    float avgIR = irSum / valid;
    float avgRed = redSum / valid;
    float glucose_concentration = BIAS + (K_IR * avgIR) + (K_RED * avgRed);
    float absolute_glucose = abs(glucose_concentration);
    Serial.print("Red:");
    Serial.print(avgRed, 0);
    Serial.print(" Infrared:");
    Serial.print(avgIR, 0);
    Serial.print(" glucose concentration:");
    Serial.println(absolute_glucose, 1);
    
  } else {
    Serial.println("Red:0 Infrared:0 glucose concentration:0");
  }
  
  delay(2000);
}
