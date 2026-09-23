#!/usr/bin/env bash
# Paper-era knob lists recovered from results/ + the submitted PDFs.
# Source after common.sh. Does not change AE defaults unless sourced.
#
# Q is queries_per_block (not Q * pipe_width). Figure 5 uses pw=2 for
# Quiver/FlashANNS and pw=1 for GustANN. The lists below are the reduced
# Fig 5 sweep: full occupancy grids were run at submission, but the plotted
# Pareto only needs these.

apply_paper_knobs() {
  export REPEAT="${REPEAT:-20}"
  export TOPK="${TOPK:-10}"
  # Force Fig 5 lists: common.sh already exported the wide AE defaults, so
  # ${VAR:-default} would not override them.
  export NUM_BLOCKS_LIST="108,216,324,432,540,648,756,864,972,1080"
  export MINI_BATCH_LIST="16,24,32,48,64,96,128,192"

  export QUIVER_LOWLAT_Q=1
  export QUIVER_LOWLAT_NB="108,216,324"
  export QUIVER_MID_Q=2
  export QUIVER_MID_NB="216,324,432,540,648,756,864,972"
  export QUIVER_DEEP_EXTRA_Q="3 4"
  export QUIVER_DEEP_EXTRA_NB="864,972"
  export FIG8_NUM_BLOCKS_LIST="108,216,324,432,540,648,756,864,972"

  export FLASH_PW_LIST=2
  export GUST_PW_LIST=1
  export QUIVER_PW_LIST=2
  export QUIVER_Q_LIST="1 2"

  export PIPE_WIDTH_QUIVER=2
  export PIPE_WIDTH_FLASH=2
  export PIPE_WIDTH_GUST=1
  export QUERIES_PER_BLOCK=2
  export PLUS_S_Q=2

  # Iso-throughput operating points, results/tech-breakdown-recall90-70k/
  export POINT_QUIVER_BLOCKS=216
  export POINT_FLASH_BLOCKS=756
  export POINT_PLUS_S_BLOCKS=324
  export POINT_GUST_MINI_BATCH=48
}
