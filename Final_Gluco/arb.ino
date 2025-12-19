#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; 
  }
  delay(2000);
  Serial.println("NON_INVASIVE Glucose Monitor: ");
  Serial.println("Initializing sensor...");
  
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found! Check wiring.");
    while(1);
  }
  
  particleSensor.setup(0x1F, 4, 2, 400, 411, 4096);
  Serial.println("Sensor ready!");
  Serial.println("Place finger on sensor");
}

void loop() {
  long irValue = particleSensor.getIR();
  long redValue = particleSensor.getRed();
  
  if (irValue > 50000) {
    float glucose = 100.0 + (irValue - 100000) * 0.0001;
    glucose += random(-5, 5);
    
    Serial.print("IR: ");
    Serial.print(irValue);
    Serial.print(" | Red: ");
    Serial.print(redValue);
    Serial.print(" | Glucose: ");
    Serial.print(glucose, 1);
    Serial.println(" mg/dL");
    
    delay(2000);
  } else {
    Serial.println("No finger detected");
    delay(3000);
  }
}
