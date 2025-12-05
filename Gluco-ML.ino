#include <Wire.h>
#include "MAX30105.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <LiquidCrystal_I2C.h>

// ================== CONFIG ==================
#define SAMPLE_WINDOW 2000
#define FEATURE_INTERVAL 5000
#define SAMPLE_RATE 100
#define BUFFER_SIZE (SAMPLE_WINDOW / 1000 * SAMPLE_RATE)

#define N_MIN_SAMPLES 20
#define ERROR_THRESHOLD_RMSE 12.0f

const float HYPERGLYCEMIA_THRESHOLD = 180.0;
const float HYPOGLYCEMIA_THRESHOLD  = 70.0;

const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";
const char* serverUrl = "http://192.168.1.10:5000/api/data";

// ================== OBJECTS ==================
MAX30105 particleSensor;
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ================== GLOBALS ==================
float irBuffer[BUFFER_SIZE];
float redBuffer[BUFFER_SIZE];
unsigned long sampleTimestamps[BUFFER_SIZE];
int bufferIndex = 0;

unsigned long lastFeatureTime = 0;
unsigned long lastSampleTime = 0;

enum SystemMode { TRAINING, PREDICTION };
SystemMode currentMode = TRAINING;

float modelCoefficients[6] = {0};
float lastGlucosePrediction = 0.0;
float lastModelRmse = -1.0;
int trainingSamplesCount = 0;

// ================== SETUP ==================
void setup() {
  Serial.begin(115200);
  Wire.begin();

  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Glucose Monitor");

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    lcd.setCursor(0,1);
    lcd.print("MAX NOT FOUND");
    while(1);
  }

  byte ledBrightness = 0x1F;
  byte sampleAverage = 4;
  byte ledMode = 2;
  int sampleRateHz = SAMPLE_RATE;
  int pulseWidth = 411;
  int adcRange = 4096;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode,
                       sampleRateHz, pulseWidth, adcRange);

  WiFi.begin(ssid, password);
  lcd.clear();
  lcd.print("WiFi Connecting");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  lcd.clear();
  lcd.print("WiFi Connected");
  delay(1500);

  lcd.clear();
  lcd.print("Mode: TRAINING");
  lcd.setCursor(0,1);
  lcd.print("Enter Glucose");
}

// ================== LOOP ==================
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

    float features[5];
    extractFeatures(features);

    if (currentMode == TRAINING)
      handleTrainingMode(features);
    else
      handlePredictionMode(features);
  }
}

// ================== FEATURE EXTRACTION ==================
void extractFeatures(float* f) {

  float irMean = 0, redMean = 0;
  float irMax = irBuffer[0], irMin = irBuffer[0];

  for (int i = 0; i < bufferIndex; i++) {
    irMean += irBuffer[i];
    redMean += redBuffer[i];
    if (irBuffer[i] > irMax) irMax = irBuffer[i];
    if (irBuffer[i] < irMin) irMin = irBuffer[i];
  }

  irMean /= bufferIndex;
  redMean /= bufferIndex;

  float ratio = irMean / (redMean + 1e-6);
  float AC = irMax - irMin;
  float DC = irMean;
  float PI_feature = AC / (DC + 1e-6);

  int maxIndex = 0, minIndex = 0;
  for (int i = 1; i < bufferIndex; i++) {
    if (irBuffer[i] > irBuffer[maxIndex]) maxIndex = i;
    if (irBuffer[i] < irBuffer[minIndex]) minIndex = i;
  }

  float slope = 0;
  float dt = sampleTimestamps[maxIndex] - sampleTimestamps[minIndex];
  if (dt > 1)
    slope = (irBuffer[maxIndex] - irBuffer[minIndex]) / dt * 1000;

  f[0] = ratio;
  f[1] = AC;
  f[2] = DC;
  f[3] = PI_feature;
  f[4] = slope;
}

// ================== TRAINING MODE ==================
void handleTrainingMode(float* f) {

  lcd.clear();
  lcd.print("TRAINING  N=");
  lcd.print(trainingSamplesCount);
  lcd.setCursor(0,1);
  lcd.print("RMSE=");
  if (lastModelRmse < 0) lcd.print("--");
  else lcd.print(lastModelRmse,1);

  if (Serial.available()) {
    float refGlucose = Serial.parseFloat();
    while (Serial.available()) Serial.read();

    sendTrainingDataToFlask(f, refGlucose);
    trainingSamplesCount++;

    if (trainingSamplesCount >= N_MIN_SAMPLES &&
        lastModelRmse > 0 &&
        lastModelRmse < ERROR_THRESHOLD_RMSE) {

      currentMode = PREDICTION;

      lcd.clear();
      lcd.print("Training OK!");
      delay(2000);
    }
  }
}

// ================== SEND TO FLASK ==================
void sendTrainingDataToFlask(float* f, float refGlucose) {

  if (WiFi.status() == WL_CONNECTED) {

    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    String payload =
      "{\"ratio\":" + String(f[0],6) +
      ",\"ac\":" + String(f[1],3) +
      ",\"dc\":" + String(f[2],2) +
      ",\"PI_feature\":" + String(f[3],6) +
      ",\"slope\":" + String(f[4],6) +
      ",\"glucose\":" + String(refGlucose,1) + "}";

    http.POST(payload);
    String resp = http.getString();
    http.end();

    int rmsePos = resp.indexOf("RMSE=");
    if (rmsePos >= 0) {
      lastModelRmse = resp.substring(rmsePos + 5).toFloat();
    }

    int coeffPos = resp.indexOf("COEFFS=");
    if (coeffPos >= 0) {
      sscanf(resp.substring(coeffPos + 7).c_str(),
        "%f,%f,%f,%f,%f,%f",
        &modelCoefficients[0], &modelCoefficients[1],
        &modelCoefficients[2], &modelCoefficients[3],
        &modelCoefficients[4], &modelCoefficients[5]
      );
    }
  }
}

// ================== PREDICTION MODE ==================
void handlePredictionMode(float* f) {

  lastGlucosePrediction =
    modelCoefficients[0] +
    modelCoefficients[1]*f[0] +
    modelCoefficients[2]*f[1] +
    modelCoefficients[3]*f[2] +
    modelCoefficients[4]*f[3] +
    modelCoefficients[5]*f[4];

  lcd.clear();
  lcd.print("Glucose:");
  lcd.setCursor(0,1);
  lcd.print(lastGlucosePrediction,1);
}
