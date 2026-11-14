# E1 / E2 results

Recordings: door_02, hall_02, room_01 (1103 test samples in total). Task: free vs blocked. Split: leave-one-recording-out. 3 seeds, 30 epochs, device mps.
Still/moving split at the median gyro rate, 0.233 rad/s. Brackets: 95% bootstrap CI over test recordings.

| Model | Motion | n | Per-zone macro F1 | Exact match | Hamming acc. |
|---|---|---|---|---|---|
| majority class (reference) | all | 1103 | 0.364 [0.227, 0.409] | 0.241 [0.087, 0.389] | 0.540 [0.297, 0.663] |
| majority class (reference) | still | 551 | 0.335 [0.263, 0.406] | 0.165 [0.074, 0.371] | 0.482 [0.355, 0.685] |
| majority class (reference) | moving | 552 | 0.386 [0.277, 0.416] | 0.317 [0.130, 0.396] | 0.599 [0.387, 0.654] |
| thermal | all | 1103 | 0.518 [0.465, 0.535] | 0.291 [0.203, 0.347] | 0.615 [0.487, 0.671] |
| thermal | still | 551 | 0.543 [0.468, 0.543] | 0.265 [0.197, 0.319] | 0.607 [0.482, 0.685] |
| thermal | moving | 552 | 0.480 [0.446, 0.503] | 0.316 [0.214, 0.358] | 0.623 [0.495, 0.718] |
| event | all | 1103 | 0.470 [0.382, 0.488] | 0.267 [0.116, 0.352] | 0.578 [0.420, 0.684] |
| event | still | 551 | 0.465 [0.347, 0.478] | 0.244 [0.112, 0.335] | 0.556 [0.383, 0.673] |
| event | moving | 552 | 0.474 [0.431, 0.509] | 0.290 [0.122, 0.411] | 0.601 [0.479, 0.718] |
| late blend (E2) | all | 1103 | 0.470 [0.397, 0.495] | 0.297 [0.148, 0.366] | 0.606 [0.428, 0.696] |
| late blend (E2) | still | 551 | 0.479 [0.364, 0.498] | 0.258 [0.122, 0.326] | 0.586 [0.389, 0.687] |
| late blend (E2) | moving | 552 | 0.457 [0.420, 0.489] | 0.336 [0.190, 0.385] | 0.626 [0.489, 0.725] |

| Model | Parameters | MFLOPs | CPU ms per update (1 thread) |
|---|---|---|---|
| thermal | 99,702 | 16.2 | 0.56 |
| event | 99,846 | 143.8 | 4.28 |
| late blend (E2) | 199,548 | 159.9 | 4.84 |

Test-label distribution (all zones): free 1987 (60%), blocked 1322 (40%)
