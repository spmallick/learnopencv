#include <ArduinoBLE.h>
#include <TensorFlowLite.h>
#include "digit_model.h"  // generated from xxd

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// BLE service/characteristics
BLEService digitService("19b10000-e8f2-537e-4f6c-d104768a1214");
BLECharacteristic imageChar("19b10001-e8f2-537e-4f6c-d104768a1214", BLEWriteWithoutResponse, 20);
BLEStringCharacteristic resultChar("19b10002-e8f2-537e-4f6c-d104768a1214", BLERead | BLENotify, 20);

uint8_t imageBuffer[64];
int bufferIndex = 0;

// TensorFlow Lite setup
const tflite::Model* model = tflite::GetModel(__digit_model_int8_tflite);
tflite::AllOpsResolver resolver;
constexpr int tensorArenaSize = 8 * 1024;  // 8 KB
uint8_t tensorArena[tensorArenaSize];

// Error reporter
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

extern "C" void DebugLog(const char* s) {
  Serial.print(s);
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // BLE
  if (!BLE.begin()) {
    Serial.println("BLE init failed!");
    while (1);
  }
  BLE.setLocalName("DigitNano");
  BLE.setAdvertisedService(digitService);
  digitService.addCharacteristic(imageChar);
  digitService.addCharacteristic(resultChar);
  BLE.addService(digitService);
  BLE.advertise();
  Serial.println("BLE advertising started...");

  // Print MAC address
  Serial.print("BLE MAC Address: ");
  Serial.println(BLE.address());

  // TensorFlow Lite
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensorArena, tensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void classifyDigit() {
  Serial.println("Running classification...");
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < 64; i++) input->data.f[i] = imageBuffer[i] / 255.0f;
  } 
  else if (input->type == kTfLiteInt8) {
    for (int i = 0; i < 64; i++) input->data.int8[i] = (int8_t)(imageBuffer[i] - 128);
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Find predicted digit
  int maxIndex = 0;
  float maxVal = -128.0f;

  if (output->type == kTfLiteFloat32) {
    maxVal = output->data.f[0];
    for (int i = 1; i < 10; i++) {
      if (output->data.f[i] > maxVal) {
        maxVal = output->data.f[i];
        maxIndex = i;
      }
    }
  } 
  else if (output->type == kTfLiteInt8) {
    for (int i = 0; i < 10; i++) {
      float val = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
      if (val > maxVal) {
        maxVal = val;
        maxIndex = i;
      }
    }
  }

  Serial.print("Predicted digit: ");
  Serial.println(maxIndex);

  // Send result via BLE
  char resultStr[10];
  sprintf(resultStr, "%d", maxIndex);
  resultChar.writeValue(resultStr);
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    while (central.connected()) {
      if (imageChar.written()) {
        int len = imageChar.valueLength();
        const uint8_t* data = imageChar.value();

        Serial.print("Received chunk of ");
        Serial.print(len);
        Serial.println(" bytes");

        for (int i = 0; i < len; i++) {
          if (bufferIndex < 64) {
            imageBuffer[bufferIndex++] = data[i];
          }
        }
        Serial.println();

        if (bufferIndex >= 64) {
          bufferIndex = 0;
          classifyDigit();
        }
      }
    }
  }
}
