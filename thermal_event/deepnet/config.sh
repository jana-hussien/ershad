# Settings for running E1 on CMU-Q's DeepNet. Every script here reads this file.
# EDIT MCS_LABEL and ENV_NAME if they are not yours (ANDREWID is your login name on the cluster).
ANDREWID=${ANDREWID:-$USER}
MCS_LABEL=sura                                    # your group's --mcs-label
ENV_NAME=daredevil                                # your conda env in /data1/$ANDREWID/.conda/envs

RUN=/data1/$ANDREWID/e1_run                       # everything lives here: code, data, results, logs
PY=/data1/$ANDREWID/.conda/envs/$ENV_NAME/bin/python
PIP=/data1/$ANDREWID/.conda/envs/$ENV_NAME/bin/pip
RCLONE=$HOME/bin/rclone
REMOTE=od                                         # the rclone remote name (from your Mac's rclone.conf)
KAGGLE_USER=janahussien90                         # owner of the dalilsense-e1-* datasets (optional, see setup.sh)

# indoor recordings with normal light (M2DGR README); door_01 is left out (outdoor -> indoor -> outdoor)
OLD="room_01 hall_02 door_02"                     # labels already in the repo; E1 inputs from Kaggle or rebuilt from the bag
NEW="room_02 lift_01 hall_03 hall_04 lift_03 room_03 lift_04 hall_01 hall_05 lift_02"   # smallest first
EVERY=2                                           # 2 samples per second (neighbours 0.25 s apart are near-identical)
MAX_BAGS=3                                        # bags on disk at once (largest is 44 GB)

# training jobs: each trains its models on all 13 recordings, 1 run, holding out one place type at a time.
# Grouped so the four jobs that run at once take similar time (estimated from last week's run).
TRAIN_GROUPS=(  # (not GROUPS: that name is built into bash)
  "images:count_40,count_80,count_120,count_240,count_40_raw,stacked_6x40,stacked_rvt_10x5,voxel_6,voxel5_norm,time_surface,time_surface_tau25,time_surface_tau100,time_surface_zubic,time_surface_tevio,fixed_n_27k,fixed_n_50k,fixed_n_200k,fixed_n_800k,thermal,thermal_x2,thermal+stacked_cnn"
  "spiking_a:snn_12x20,snn_voxel5"
  "spiking_b:thermal+snn,graph"
  "aegnn:aegnn"
  "eventmg:eventmg_style"
)
