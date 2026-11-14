# E1 / E2 results

Recordings: door_02, hall_02, room_01 (1103 test samples in total). Task: clear/person/furniture/other. Split: leave-one-recording-out. 3 seeds, 30 epochs, device mps.
Still/moving split at the median gyro rate, 0.233 rad/s. Brackets: 95% bootstrap CI over test recordings.

| Model | Motion | n | Per-zone macro F1 | Exact match | Hamming acc. |
|---|---|---|---|---|---|
| majority class (reference) | all | 1103 | 0.188 [0.114, 0.243] | 0.346 [0.138, 0.475] | 0.600 [0.297, 0.788] |
| majority class (reference) | still | 551 | 0.183 [0.133, 0.239] | 0.312 [0.189, 0.426] | 0.576 [0.364, 0.768] |
| majority class (reference) | moving | 552 | 0.192 [0.139, 0.256] | 0.380 [0.170, 0.638] | 0.625 [0.387, 0.855] |
| thermal | all | 1103 | 0.226 [0.151, 0.312] | 0.279 [0.134, 0.336] | 0.564 [0.293, 0.698] |
| thermal | still | 551 | 0.237 [0.146, 0.312] | 0.239 [0.118, 0.308] | 0.535 [0.253, 0.674] |
| thermal | moving | 552 | 0.209 [0.148, 0.251] | 0.319 [0.161, 0.430] | 0.593 [0.355, 0.776] |
| event | all | 1103 | 0.243 [0.174, 0.243] | 0.219 [0.043, 0.304] | 0.504 [0.253, 0.603] |
| event | still | 551 | 0.243 [0.167, 0.244] | 0.189 [0.034, 0.282] | 0.473 [0.240, 0.609] |
| event | moving | 552 | 0.240 [0.183, 0.247] | 0.249 [0.057, 0.312] | 0.534 [0.274, 0.601] |
| late blend (E2) | all | 1103 | 0.224 [0.163, 0.270] | 0.279 [0.095, 0.350] | 0.568 [0.288, 0.701] |
| late blend (E2) | still | 551 | 0.236 [0.160, 0.273] | 0.246 [0.092, 0.322] | 0.545 [0.260, 0.690] |
| late blend (E2) | moving | 552 | 0.206 [0.162, 0.238] | 0.312 [0.101, 0.382] | 0.591 [0.331, 0.738] |

| Model | Parameters | MFLOPs | CPU ms per update (1 thread) |
|---|---|---|---|
| thermal | 102,012 | 16.2 | 0.97 |
| event | 102,156 | 143.8 | 8.89 |
| late blend (E2) | 204,168 | 159.9 | 9.85 |

Test-label distribution (all zones): clear 1987 (60%), person 659 (20%), furniture 169 (5%), other 494 (15%)
