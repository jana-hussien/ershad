# Settings for running E1 on CMU-Q's DeepNet. Every script here reads this file.
# Nothing to edit normally: the Andrew ID is your cluster login, and the conda env is the first one found
# in /data1/<andrewid>/.conda/envs. Set ENV_NAME / MCS_LABEL here only to override them.
ANDREWID=${ANDREWID:-$USER}
MCS_LABEL=${MCS_LABEL:-sura}                      # the course's --mcs-label
if [ -z "${ENV_NAME:-}" ]; then
  for e in /data1/$ANDREWID/.conda/envs/*/bin/python; do [ -x "$e" ] && { ENV_NAME=$(basename "$(dirname "$(dirname "$e")")"); break; }; done
fi
ENV_NAME=${ENV_NAME:-NONE}

RUN=/data1/$ANDREWID/e1_run                       # everything lives here: code, data, results, logs
PY=/data1/$ANDREWID/.conda/envs/$ENV_NAME/bin/python
PIP=/data1/$ANDREWID/.conda/envs/$ENV_NAME/bin/pip
RCLONE=$HOME/bin/rclone
REMOTE=od                                         # the rclone remote name (from your Mac's rclone.conf)

# indoor recordings with normal light (M2DGR README). door_01 and door_02 start (and door_01 ends) outdoors:
# only their indoor part is used (keep/door_01.json; door_02's keep.json is in the repo cache)
OLD="room_01 hall_02 door_02"                     # labels already in the repo; E1 inputs rebuilt from the bag
NEW="room_02 lift_01 hall_03 hall_04 lift_03 room_03 lift_04 hall_01 hall_05 door_01 lift_02"   # smallest first
EVERY=1                                           # every sample (4 per second): neighbours differ in labels 26-47% of the time
MAX_BAGS=3                                        # bags on disk at once (largest is 44 GB)

# training jobs: each trains its models on all 14 recordings, 1 run, holding out one place type at a time.
# Grouped so the four jobs that run at once take similar time (estimated from last week's run).
TRAIN_GROUPS=(  # (not GROUPS: that name is built into bash)
  "images_a:count_40,count_80,count_120,count_240,count_40_raw,stacked_6x40,stacked_rvt_10x5,voxel_6,voxel5_norm,thermal,thermal_x2"
  "images_b:time_surface,time_surface_tau25,time_surface_tau100,time_surface_zubic,time_surface_tevio,fixed_n_27k,fixed_n_50k,fixed_n_200k,fixed_n_800k,thermal+stacked_cnn"
  "spiking_a:snn_12x20,snn_voxel5"
  "spiking_b:thermal+snn,graph"
  "aegnn:aegnn"
  "eventmg:eventmg_style"
)
