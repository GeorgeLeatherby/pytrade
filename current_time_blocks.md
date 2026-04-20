============================================================
CREATING TIME SERIES BLOCKS
============================================================
Episode requirements:
  Lookback window: 0 days
  Episode length: 252 days
  Min episode requirement: 252 days
  Buffer multiplier: 1.0x
  Min viable train block: 252.0 days
  Min viable validation block: 252 days
  Min timeframe for validation: 1260 days
  Min timeframe for training: 315 days
  Required timeframe size: 1260 days
  Total available days: 5,834
  Target validation ratio: 20.0%
  Possible timeframes: 4 (based on integer division)

Timeframe 1:
  Training block: train_00
    Date range: 2002-05-02 to 2006-05-31
    Days: 1,008
    Episodes: 4
    Episode start range: [0, 756]
  Validation block: val_00
    Date range: 2006-06-01 to 2007-06-05
    Days: 252
    Episodes: 1
    Episode start range: [1008, 1008]

Timeframe 2:
  Training block: train_01
    Date range: 2007-06-06 to 2011-06-24
    Days: 1,008
    Episodes: 4
    Episode start range: [1260, 2016]
  Validation block: val_01
    Date range: 2011-06-27 to 2012-07-03
    Days: 252
    Episodes: 1
    Episode start range: [2268, 2268]

Timeframe 3:
  Training block: train_02
    Date range: 2012-07-06 to 2016-08-12
    Days: 1,008
    Episodes: 4
    Episode start range: [2520, 3276]
  Validation block: val_02
    Date range: 2016-08-15 to 2017-08-18
    Days: 252
    Episodes: 1
    Episode start range: [3528, 3528]

Timeframe 4:
  Training block: train_03
    Date range: 2017-08-21 to 2024-03-01
    Days: 1,643
    Episodes: 6
    Episode start range: [3780, 5171]
  Validation block: val_03
    Date range: 2024-03-04 to 2025-10-21
    Days: 411
    Episodes: 1
    Episode start range: [5423, 5582]

Sampling weights calculated:
  Training weights: [0.21598457 0.21598457 0.21598457 0.35204628]
  Validation weights: [0.2159383 0.2159383 0.2159383 0.3521851]

============================================================
TIME SERIES SPLITTING COMPLETE
============================================================
Training blocks: 4
Validation blocks: 4
Total training episodes: 18
Total validation episodes: 4
Actual validation ratio: 18.2%
============================================================
