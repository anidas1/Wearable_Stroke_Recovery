// sampler_features.ino
// 3-channel RAW EMG, GPIO4 + GPIO7 + GPIO10
// Output: one CSV line per 100ms window
// Format: timestamp_us,f0..f23,prediction

#include <Arduino.h>
#include <math.h>

// ─────────────────────────────────────────────────────────────────────
// PIN CONFIG
// ─────────────────────────────────────────────────────────────────────
#define RAW1_PIN    4
#define RAW2_PIN    7
#define RAW3_PIN    10
#define SIGNAL_PIN  17

// ─────────────────────────────────────────────────────────────────────
// SAMPLING CONFIG
// ─────────────────────────────────────────────────────────────────────
#define SAMPLE_RATE_HZ 2000
#define WINDOW_SIZE    200
#define NUM_CHANNELS   3
#define NUM_FEATURES   8
#define TOTAL_FEATURES (NUM_CHANNELS * NUM_FEATURES)

// ─────────────────────────────────────────────────────────────────────
// FEATURE THRESHOLDS
// ─────────────────────────────────────────────────────────────────────
#define WAMP_THRESH  30.0f
#define ZC_THRESH    30.0f
#define SSC_THRESH   30.0f

// ─────────────────────────────────────────────────────────────────────
// TIMER
// ─────────────────────────────────────────────────────────────────────
hw_timer_t*   timer       = NULL;
volatile bool sampleReady = false;
void IRAM_ATTR onTimer()  { sampleReady = true; }

// ─────────────────────────────────────────────────────────────────────
// WINDOW BUFFER
// ─────────────────────────────────────────────────────────────────────
float buf[NUM_CHANNELS][WINDOW_SIZE];
int      bufIdx      = 0;
bool     bufFull     = false;
uint32_t windowStart = 0;  // timestamp of first sample in window

// ─────────────────────────────────────────────────────────────────────
// FEATURE EXTRACTION
// ─────────────────────────────────────────────────────────────────────
void extractFeatures(const float* w, int len, float* out) {
  float dc = 0;
  for (int i = 0; i < len; i++) dc += w[i];
  dc /= len;

  float c[200];
  for (int i = 0; i < len; i++) c[i] = w[i] - dc;

  float sum_abs = 0;
  for (int i = 0; i < len; i++) sum_abs += fabsf(c[i]);
  out[0] = sum_abs / len;

  float sum_sq = 0;
  for (int i = 0; i < len; i++) sum_sq += c[i] * c[i];
  out[1] = sqrtf(sum_sq / len);

  float var = 0;
  for (int i = 0; i < len; i++) var += c[i] * c[i];
  out[2] = var / (len - 1);

  float peak = 0;
  for (int i = 0; i < len; i++) { float a = fabsf(c[i]); if (a > peak) peak = a; }
  out[3] = peak;

  float wl = 0;
  for (int i = 1; i < len; i++) wl += fabsf(c[i] - c[i-1]);
  out[4] = wl;

  float wamp = 0;
  for (int i = 1; i < len; i++)
    if (fabsf(c[i] - c[i-1]) >= WAMP_THRESH) wamp++;
  out[5] = wamp;

  float zc = 0;
  for (int i = 1; i < len; i++)
    if (((c[i] > 0 && c[i-1] < 0) || (c[i] < 0 && c[i-1] > 0)) &&
        fabsf(c[i] - c[i-1]) >= ZC_THRESH) zc++;
  out[6] = zc;

  float ssc = 0;
  for (int i = 1; i < len - 1; i++) {
    float d1 = c[i] - c[i-1], d2 = c[i+1] - c[i];
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        (fabsf(d1) >= SSC_THRESH || fabsf(d2) >= SSC_THRESH)) ssc++;
  }
  out[7] = ssc;
}

void extractAllChannels(const float b[][WINDOW_SIZE], float* feat) {
  for (int ch = 0; ch < NUM_CHANNELS; ch++)
    extractFeatures(b[ch], WINDOW_SIZE, feat + ch * NUM_FEATURES);
}

// ─────────────────────────────────────────────────────────────────────
// LDA PARAMS
// ─────────────────────────────────────────────────────────────────────
float scaler_mean[24] = {22.626900f, 29.576692f, 1544.471361f, 91.252080f, 1167.491489f, 3.907801f, 0.968794f, 0.892199f, 17.575877f, 23.081752f, 1013.230471f, 76.189855f, 1457.870922f, 10.195035f, 3.076596f, 2.134043f, 19.396503f, 25.692882f, 1497.093461f, 80.541954f, 1095.177305f, 2.808511f, 0.769504f, 1.254610f};
float scaler_std[24]  = {19.578175f, 25.728747f, 3447.050763f, 78.187795f, 771.215620f, 6.322425f, 1.418377f, 1.364082f, 17.267382f, 21.803600f, 1599.659946f, 62.821110f, 1238.813202f, 15.825497f, 4.207198f, 2.730400f, 21.522233f, 28.800761f, 4123.173745f, 84.806580f, 573.829983f, 5.927236f, 1.132401f, 1.745429f};
float lda_coef[1][24] = {
  {2.919325f, -4.488007f, 0.970437f, 0.289419f, 3.309649f, -1.041856f, 0.206585f, 0.036315f, -0.016016f, -2.158030f, 0.862738f, -0.595987f, 0.803113f, 1.231650f, -0.225502f, -0.191591f, -1.121562f, 1.161586f, -0.447634f, 0.086603f, -0.543332f, 0.748518f, 0.027324f, -0.087960f},
};
float lda_intercept[1] = {-0.497801f};

// ─────────────────────────────────────────────────────────────────────
// LDA INFERENCE
// ─────────────────────────────────────────────────────────────────────
int ldaPredict(float* feat) {
  float x[24];
  for (int i = 0; i < 24; i++)
    x[i] = (feat[i] - scaler_mean[i]) / scaler_std[i];
  float score = lda_intercept[0];
  for (int i = 0; i < 24; i++)
    score += lda_coef[0][i] * x[i];
  return (score > 0) ? 1 : 0;
}

// ─────────────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(921600);
  delay(500);

  pinMode(SIGNAL_PIN, OUTPUT);
  digitalWrite(SIGNAL_PIN, LOW);

  analogReadResolution(12);
  analogSetPinAttenuation(RAW1_PIN, ADC_11db);
  analogSetPinAttenuation(RAW2_PIN, ADC_11db);
  analogSetPinAttenuation(RAW3_PIN, ADC_11db);
  pinMode(RAW1_PIN, INPUT);
  pinMode(RAW2_PIN, INPUT);
  pinMode(RAW3_PIN, INPUT);

  timer = timerBegin(SAMPLE_RATE_HZ);
  timerAttachInterrupt(timer, &onTimer);
  timerAlarm(timer, 1, true, 0);

  // Print CSV header
  Serial.print("timestamp_us");
  for (int i = 0; i < TOTAL_FEATURES; i++) Serial.printf(",f%d", i);
  Serial.println(",prediction");
}

// ─────────────────────────────────────────────────────────────────────
// LOOP
// ─────────────────────────────────────────────────────────────────────
void loop() {
  if (!sampleReady) return;
  sampleReady = false;

  float s1 = analogRead(RAW1_PIN);
  float s2 = analogRead(RAW2_PIN);
  float s3 = analogRead(RAW3_PIN);

  // Record timestamp of first sample in window
  if (bufIdx == 0) windowStart = micros();

  buf[0][bufIdx] = s1;
  buf[1][bufIdx] = s2;
  buf[2][bufIdx] = s3;
  bufIdx++;

  if (bufIdx >= WINDOW_SIZE) { bufIdx = 0; bufFull = true; }
  if (!bufFull) return;
  bufFull = false;

  float feat[TOTAL_FEATURES];
  extractAllChannels(buf, feat);

  int pred = ldaPredict(feat);

  // Motor trigger
  digitalWrite(SIGNAL_PIN, pred == 1 ? HIGH : LOW);

  // Single CSV line: timestamp, features, prediction
  Serial.print(windowStart);
  for (int i = 0; i < TOTAL_FEATURES; i++) {
    Serial.print(",");
    Serial.print(feat[i], 4);
  }
  Serial.println(pred == 1 ? ",REACH" : ",REST");
}
