#include <Wire.h>
#include "MAX30105.h"
// doc https://learn.sparkfun.com/tutorials/max30105-particle-and-pulse-ox-sensor-hookup-guide/all
MAX30105 particleSensor;

void setup()
{
  Wire.begin(); 
  Serial.begin(115200);
  
  Serial.println("");
  Serial.println("Initializing...");
 
  // Initialize sensor
  while(!particleSensor.begin(Wire, I2C_SPEED_FAST)){
    Serial.print(".");
  }
  Serial.println("OK!");
  
 
  //Setup to sense a nice looking saw tooth on the plotter
  byte ledBrightness = 0x1F; //Options: 0=Off to 255=50mA
  byte sampleAverage = 4; //Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 1; //Options: 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
  int sampleRate = 800; //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
  int pulseWidth = 411; //Options: 69, 118, 215, 411
  int adcRange = 4096; //Options: 2048, 4096, 8192, 16384
 
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange); //Configure sensor with these settings
 
  Serial.println("red");  
}
 
 
void loop()
{
//  uint32_t sensor_ir = particleSensor.getIR();
  uint32_t sensor_red = particleSensor.getRed();

  Serial.println(sensor_red);
}
