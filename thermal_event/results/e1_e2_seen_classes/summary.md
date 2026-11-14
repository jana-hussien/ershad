# E1 / E2 results

Recordings: door_02, room_01 (589 test samples in total). Task: clear/person/furniture/other. Split: seen places: 5 held-out time blocks per recording, 1 s gap. 2 seeds, 20 epochs, device mps.
Still/moving split at the median gyro rate, 0.164 rad/s. Brackets: 95% bootstrap CI over test blocks.

| Model | Motion | n | Per-zone macro F1 | Exact match | Hamming acc. |
|---|---|---|---|---|---|
| majority class (reference) | all | 589 | 0.177 [0.132, 0.207] | 0.309 [0.159, 0.471] | 0.546 [0.361, 0.709] |
| majority class (reference) | still | 294 | 0.163 [0.113, 0.198] | 0.221 [0.123, 0.372] | 0.484 [0.288, 0.655] |
| majority class (reference) | moving | 295 | 0.189 [0.145, 0.230] | 0.397 [0.191, 0.594] | 0.608 [0.410, 0.768] |
| thermal | all | 589 | 0.518 [0.405, 0.557] | 0.350 [0.221, 0.467] | 0.688 [0.589, 0.771] |
| thermal | still | 294 | 0.547 [0.403, 0.635] | 0.282 [0.161, 0.455] | 0.680 [0.597, 0.779] |
| thermal | moving | 295 | 0.464 [0.384, 0.502] | 0.417 [0.268, 0.564] | 0.695 [0.575, 0.802] |
| event | all | 589 | 0.368 [0.292, 0.402] | 0.278 [0.164, 0.407] | 0.564 [0.431, 0.678] |
| event | still | 294 | 0.334 [0.259, 0.380] | 0.185 [0.111, 0.273] | 0.526 [0.389, 0.623] |
| event | moving | 295 | 0.402 [0.308, 0.445] | 0.369 [0.203, 0.555] | 0.602 [0.455, 0.752] |
| late blend (E2) | all | 589 | 0.478 [0.372, 0.513] | 0.348 [0.222, 0.472] | 0.681 [0.568, 0.773] |
| late blend (E2) | still | 294 | 0.497 [0.395, 0.561] | 0.287 [0.175, 0.442] | 0.676 [0.572, 0.775] |
| late blend (E2) | moving | 295 | 0.432 [0.348, 0.481] | 0.408 [0.243, 0.571] | 0.686 [0.541, 0.805] |

| Model | Parameters | MFLOPs | CPU ms per update (1 thread) |
|---|---|---|---|
| thermal | 102,012 | 16.2 | 0.53 |
| event | 102,156 | 143.8 | 4.48 |
| late blend (E2) | 204,168 | 159.9 | 5.01 |

Test-label distribution (all zones): clear 965 (55%), person 512 (29%), furniture 70 (4%), other 220 (12%)
