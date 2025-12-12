#include <Wire.h>
#include "MAX30105.h"
#include <LiquidCrystal_I2C.h>
#include <math.h>

#define SAMPLE_WINDOW_MS 2000
#define FEATURE_INTERVAL 5000
#define SAMPLE_RATE 100
#define BUFFER_SIZE (SAMPLE_WINDOW_MS / 1000 * SAMPLE_RATE)

#define WINDOWS_PER_SAMPLE 3
#define FEATURE_COUNT 5

MAX30105 particleSensor;
LiquidCrystal_I2C lcd(0x27, 16, 2);

float irBuffer[BUFFER_SIZE];
float redBuffer[BUFFER_SIZE];
unsigned long sampleTimestamps[BUFFER_SIZE];
int bufferIndex = 0;

unsigned long lastFeatureTime = 0;
unsigned long lastSampleTime = 0;

float windowFeatures[WINDOWS_PER_SAMPLE][FEATURE_COUNT];
int windowCount = 0;

//ML MODEL Coeffs
float b0 = 23.657853;
float b1 = -22.472835;
float b2 = -0.025231;
float b3 = 0.000040;
float b4 = 2618.109262;
float b5 = 4.860547;


void setup() {
  Serial.begin(115200);
  Wire.begin();

  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.print("Glucose Logger");

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    lcd.setCursor(0, 1);
    lcd.print("MAX30102 ERR");
    while (1);
  }

  byte ledBrightness = 0x1F;
  byte sampleAverage = 4;
  byte ledMode = 2;              // RED + IR
  int sampleRateHz = SAMPLE_RATE;
  int pulseWidth = 411;
  int adcRange = 4096;

  particleSensor.setup(
    ledBrightness,
    sampleAverage,
    ledMode,
    sampleRateHz,
    pulseWidth,
    adcRange
  );

  delay(1500);
  lcd.clear();
  lcd.print("Place Finger");
  lcd.setCursor(0, 1);
  lcd.print("Hold 15 sec");
}

void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime >= (1000 / SAMPLE_RATE)) {
    lastSampleTime = now;

    irBuffer[bufferIndex] = particleSensor.getIR();
    redBuffer[bufferIndex] = particleSensor.getRed();
    sampleTimestamps[bufferIndex] = now;
    bufferIndex++;

    if (bufferIndex >= BUFFER_SIZE) {
      for (int i = 0; i < BUFFER_SIZE - 1; i++) {
        irBuffer[i] = irBuffer[i + 1];
        redBuffer[i] = redBuffer[i + 1];
        sampleTimestamps[i] = sampleTimestamps[i + 1];
      }
      bufferIndex = BUFFER_SIZE - 1;
    }
  }

  if (now - lastFeatureTime >= FEATURE_INTERVAL) {
    lastFeatureTime = now;
    float features[FEATURE_COUNT];
    extractFeatures(features);
    for (int i = 0; i < FEATURE_COUNT; i++)
      windowFeatures[windowCount][i] = features[i];
    windowCount++;
    if (windowCount >= WINDOWS_PER_SAMPLE) {
      float avg[FEATURE_COUNT] = {0};
      for (int w = 0; w < WINDOWS_PER_SAMPLE; w++) {
        for (int i = 0; i < FEATURE_COUNT; i++) {
          avg[i] += windowFeatures[w][i];
        }
      }

      for (int i = 0; i < FEATURE_COUNT; i++)
        avg[i] /= WINDOWS_PER_SAMPLE;

      float ratio_avg = avg[0];
      float ac_avg    = avg[1];
      float dc_avg    = avg[2];
      float pi_avg    = avg[3];
      float slope_avg = avg[4];

      float glucose =
          b0 +
          b1 * ratio_avg +
          b2 * ac_avg +
          b3 * dc_avg +
          b4 * pi_avg +
          b5 * slope_avg;

      Serial.println("15s AVERAGED FEATURES:");

      Serial.print("ratio: ");
      Serial.println(ratio_avg, 6);

      Serial.print("ac: ");
      Serial.println(ac_avg, 3);

      Serial.print("dc: ");
      Serial.println(dc_avg, 3);

      Serial.print("pi: ");
      Serial.println(pi_avg, 6);

      Serial.print("slope: ");
      Serial.println(slope_avg, 6);

      Serial.println("--------------------------------");

      Serial.print("GLUCOSE (mg/dL): ");
      Serial.println(glucose, 2);
      Serial.println();

      lcd.clear();
      lcd.print("Glucose:");
      lcd.setCursor(0, 1);
      lcd.print(glucose, 1);

      windowCount = 0;
    }
  }
}

void extractFeatures(float* f) {
  float irMean = 0, redMean = 0;
  for (int i = 0; i < bufferIndex; i++) {
    irMean += irBuffer[i];
    redMean += redBuffer[i];
  }
  irMean /= bufferIndex;
  redMean /= bufferIndex;
  float ratio = irMean / (redMean + 1e-6);
  float acSum = 0;
  for (int i = 0; i < bufferIndex; i++) {
    float x = irBuffer[i] - irMean;
    acSum += x * x;
  }
  float AC = sqrt(acSum / bufferIndex);
  float DC = irMean;
  float PI_feature = AC / (DC + 1e-6);

  //LINEAR REGRESSION SLOPE
  float sumT = 0, sumY = 0, sumTT = 0, sumTY = 0;
  for (int i = 0; i < bufferIndex; i++) {
    float t = (float)sampleTimestamps[i];
    float y = irBuffer[i];
    sumT += t;
    sumY += y;
    sumTT += t * t;
    sumTY += t * y;
  }

  float denom = bufferIndex * sumTT - sumT * sumT;
  float slope = 0;

  if (denom != 0)
    slope = (bufferIndex * sumTY - sumT * sumY) / denom;

  f[0] = ratio;
  f[1] = AC;
  f[2] = DC;
  f[3] = PI_feature;
  f[4] = slope;
}