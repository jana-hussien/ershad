# E1 / E2 results

Recordings: door_02, room_01 (589 test samples in total). Task: free vs blocked. Split: leave-one-recording-out. 3 seeds, 30 epochs, device mps.
Still/moving split at the median gyro rate, 0.164 rad/s. Brackets: 95% bootstrap CI over test recordings.

| Model | Motion | n | Per-zone macro F1 | Exact match | Hamming acc. |
|---|---|---|---|---|---|
| majority class (reference) | all | 589 | 0.252 [0.173, 0.252] | 0.080 [0.023, 0.138] | 0.254 [0.212, 0.297] |
| majority class (reference) | still | 294 | 0.248 [0.179, 0.248] | 0.075 [0.033, 0.121] | 0.249 [0.222, 0.275] |
| majority class (reference) | moving | 295 | 0.247 [0.127, 0.267] | 0.085 [0.014, 0.154] | 0.258 [0.146, 0.367] |
| thermal | all | 589 | 0.427 [0.389, 0.455] | 0.146 [0.125, 0.168] | 0.449 [0.411, 0.486] |
| thermal | still | 294 | 0.409 [0.339, 0.451] | 0.133 [0.124, 0.142] | 0.416 [0.351, 0.476] |
| thermal | moving | 295 | 0.432 [0.428, 0.432] | 0.159 [0.126, 0.192] | 0.481 [0.467, 0.496] |
| event | all | 589 | 0.422 [0.325, 0.482] | 0.123 [0.051, 0.194] | 0.468 [0.428, 0.506] |
| event | still | 294 | 0.361 [0.299, 0.377] | 0.065 [0.038, 0.089] | 0.388 [0.386, 0.389] |
| event | moving | 295 | 0.435 [0.343, 0.495] | 0.182 [0.063, 0.304] | 0.547 [0.465, 0.632] |
| late blend (E2) | all | 589 | 0.412 [0.345, 0.464] | 0.122 [0.108, 0.136] | 0.449 [0.410, 0.487] |
| late blend (E2) | still | 294 | 0.361 [0.311, 0.376] | 0.077 [0.063, 0.092] | 0.380 [0.362, 0.397] |
| late blend (E2) | moving | 295 | 0.426 [0.371, 0.485] | 0.167 [0.123, 0.212] | 0.518 [0.456, 0.581] |

| Model | Parameters | MFLOPs | CPU ms per update (1 thread) |
|---|---|---|---|
| thermal | 99,702 | 16.2 | 0.48 |
| event | 99,846 | 143.8 | 4.21 |
| late blend (E2) | 199,548 | 159.9 | 4.69 |

Test-label distribution (all zones): free 965 (55%), blocked 802 (45%)
