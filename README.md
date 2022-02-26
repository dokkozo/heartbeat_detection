# heartbeat_detection

Detect heartbeat and calculate BPM with python using sensor connected to Arduino Uno.

## Installation
- ArduinoUno
    - Connect MAX30102 heartbeat sensor correctly (Vcc-5V, GND-GND, SDA-A4, SCL-A5)
    - Connect Arduino to PC via USB
    - Compile and write pulsesend.ino using Arduino IDE.
        - Requires MAX30105 library. Please install using library manager in Arduino IDE.
- PC
    - Windows10
    - python 3.9.7
    - Please install imported libraries to run python script.

## Directories and Files
### pulsesend/pulsesend.ino
- Code for Arduino Uno to read MAX30102 sensor data and to send sensor the data to PC using serial.

### send_bpm.py
- Python script to buffer sensor data and analyze the buffered data.
- This code may be compatible with other heartbeat sensor than MAX30102 as long as it can receive sensor value in integer, via serial port, in fixed interval and with enough sampling rate and signal-to-noise ratio.
