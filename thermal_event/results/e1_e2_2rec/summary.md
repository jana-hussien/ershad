# E1 / E2 results

Recordings: door_02, room_01 (589 test samples in total). Split: leave-one-recording-out. 3 seeds, 30 epochs, device mps.
Still/moving split at the median gyro rate, 0.164 rad/s. Brackets: 95% bootstrap CI over test recordings.

| Model | Motion | n | Per-zone macro F1 | Exact match | Hamming acc. |
|---|---|---|---|---|---|
| majority class (reference) | all | 589 | 0.097 [0.048, 0.114] | 0.068 [0.000, 0.138] | 0.191 [0.089, 0.297] |
| majority class (reference) | still | 294 | 0.088 [0.058, 0.089] | 0.058 [0.000, 0.121] | 0.167 [0.115, 0.222] |
| majority class (reference) | moving | 295 | 0.102 [0.034, 0.133] | 0.078 [0.000, 0.154] | 0.216 [0.062, 0.367] |
| thermal | all | 589 | 0.160 [0.128, 0.194] | 0.084 [0.031, 0.139] | 0.334 [0.300, 0.367] |
| thermal | still | 294 | 0.148 [0.106, 0.182] | 0.063 [0.011, 0.121] | 0.271 [0.229, 0.309] |
| thermal | moving | 295 | 0.162 [0.143, 0.195] | 0.105 [0.053, 0.157] | 0.397 [0.368, 0.428] |
| event | all | 589 | 0.167 [0.122, 0.185] | 0.037 [0.030, 0.045] | 0.306 [0.254, 0.359] |
| event | still | 294 | 0.141 [0.076, 0.179] | 0.034 [0.024, 0.045] | 0.218 [0.105, 0.340] |
| event | moving | 295 | 0.184 [0.155, 0.184] | 0.041 [0.037, 0.045] | 0.394 [0.377, 0.410] |
| late blend (E2) | all | 589 | 0.153 [0.150, 0.153] | 0.068 [0.010, 0.129] | 0.323 [0.316, 0.331] |
| late blend (E2) | still | 294 | 0.125 [0.114, 0.130] | 0.058 [0.007, 0.113] | 0.210 [0.162, 0.262] |
| late blend (E2) | moving | 295 | 0.168 [0.164, 0.168] | 0.079 [0.014, 0.143] | 0.436 [0.396, 0.477] |

| Model | Parameters | MFLOPs | CPU ms per update (1 thread) |
|---|---|---|---|
| thermal | 102,012 | 16.2 | 0.55 |
| event | 102,156 | 143.8 | 4.40 |
| late blend (E2) | 204,168 | 159.9 | 4.95 |

Test-label distribution (all zones): clear 965 (55%), person 512 (29%), furniture 70 (4%), other 220 (12%)
