# E1 / E2 results

Recordings: door_02, hall_02, room_01 (1103 test samples in total). Task: free vs blocked. Split: seen places: 5 held-out time blocks per recording, 1 s gap. 1 seeds, 20 epochs, device mps.
Still/moving split at the median gyro rate, 0.233 rad/s. Brackets: 95% bootstrap CI over test blocks.

| Model | Motion | n | Per-zone macro F1 | Exact match | Hamming acc. |
|---|---|---|---|---|---|
| majority class (reference) | all | 1103 | 0.375 [0.324, 0.418] | 0.346 [0.216, 0.494] | 0.600 [0.479, 0.719] |
| majority class (reference) | still | 551 | 0.365 [0.296, 0.417] | 0.312 [0.181, 0.461] | 0.576 [0.423, 0.717] |
| majority class (reference) | moving | 552 | 0.384 [0.331, 0.425] | 0.380 [0.216, 0.548] | 0.625 [0.497, 0.739] |
| thermal | all | 1103 | 0.738 [0.671, 0.784] | 0.456 [0.376, 0.538] | 0.751 [0.704, 0.798] |
| thermal | still | 551 | 0.727 [0.631, 0.792] | 0.432 [0.342, 0.535] | 0.739 [0.675, 0.801] |
| thermal | moving | 552 | 0.749 [0.693, 0.791] | 0.480 [0.377, 0.574] | 0.763 [0.711, 0.809] |
| event | all | 1103 | 0.656 [0.590, 0.706] | 0.377 [0.290, 0.468] | 0.670 [0.618, 0.721] |
| event | still | 551 | 0.655 [0.536, 0.730] | 0.374 [0.266, 0.492] | 0.668 [0.597, 0.738] |
| event | moving | 552 | 0.657 [0.612, 0.700] | 0.380 [0.287, 0.480] | 0.671 [0.625, 0.718] |
| late blend (E2) | all | 1103 | 0.734 [0.668, 0.786] | 0.468 [0.372, 0.564] | 0.745 [0.688, 0.799] |
| late blend (E2) | still | 551 | 0.733 [0.630, 0.811] | 0.454 [0.347, 0.584] | 0.743 [0.676, 0.816] |
| late blend (E2) | moving | 552 | 0.733 [0.663, 0.780] | 0.482 [0.362, 0.587] | 0.746 [0.670, 0.807] |

| Model | Parameters | MFLOPs | CPU ms per update (1 thread) |
|---|---|---|---|
| thermal | 99,702 | 16.2 | 1.05 |
| event | 99,846 | 143.8 | 5.01 |
| late blend (E2) | 199,548 | 159.9 | 6.06 |

Test-label distribution (all zones): free 1987 (60%), blocked 1322 (40%)
