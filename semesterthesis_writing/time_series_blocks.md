# New time blocks:

Episode requirements:
  Lookback window: 0 days
  Episode length: 252 days
  Min episode requirement: 252 days
  Buffer multiplier: 1.0x
  Min viable train block: 252 days
  Min viable validation block: 252 days
  Min super-block for validation feasibility: 1260 days
  Min super-block for training feasibility: 315 days
  Chosen minimum super-block size: 1260 days
  Purge length (config): 60 days
  Minimum train+purge+validation cycle size: 1320 days
  OOS test days at range end: 252
  Tail holdout (purge+test): 312 days
  Total available days: 6,344
  Target validation ratio (train_val_split_ratio): 20.0%
  Available days for train/val cycles: 6032
  Possible full train/purge/val cycles: 4
  Total super-block days (train+validation, no purge): 5792
  Achieved validation share over super-block subset: 19.99%

Cycle 1:
  Training block: train_00
    Date range: 2000-12-12 to 2005-09-20
    Days: 1,159
    Episodes: 4
    Episode start range: [0, 907]
    Super-block size (train+val): 1448
  Purge window: train->val_00
    Index range: [1159, 1218] (60 days)
    Date range: 2005-09-21 to 2005-12-15
  Validation block: val_00
    Date range: 2005-12-16 to 2007-02-14
    Days: 289
    Episodes: 1
    Episode start range: [1219, 1256]

Cycle 2:
  Training block: train_01
    Date range: 2007-02-15 to 2011-10-13
    Days: 1,159
    Episodes: 4
    Episode start range: [1508, 2415]
    Super-block size (train+val): 1448
  Purge window: train->val_01
    Index range: [2667, 2726] (60 days)
    Date range: 2011-10-14 to 2012-01-11
  Validation block: val_01
    Date range: 2012-01-12 to 2013-03-20
    Days: 289
    Episodes: 1
    Episode start range: [2727, 2764]

Cycle 3:
  Training block: train_02
    Date range: 2013-03-21 to 2017-11-28
    Days: 1,158
    Episodes: 4
    Episode start range: [3016, 3922]
    Super-block size (train+val): 1448
  Purge window: train->val_02
    Index range: [4174, 4233] (60 days)
    Date range: 2017-11-29 to 2018-02-26
  Validation block: val_02
    Date range: 2018-02-27 to 2019-04-23
    Days: 290
    Episodes: 1
    Episode start range: [4234, 4272]

Cycle 4:
  Training block: train_03
    Date range: 2019-04-24 to 2023-11-27
    Days: 1,158
    Episodes: 4
    Episode start range: [4524, 5430]
    Super-block size (train+val): 1448
  Purge window: train->val_03
    Index range: [5682, 5741] (60 days)
    Date range: 2023-11-28 to 2024-02-23
  Validation block: val_03
    Date range: 2024-02-26 to 2025-04-22
    Days: 290
    Episodes: 1
    Episode start range: [5742, 5780]

Final OOS holdout:
  Purge window: last_validation->test_00
    Index range: [6032, 6091] (60 days)
    Date range: 2025-04-23 to 2025-07-18
  Test block: test_00
    Date range: 2025-07-21 to 2026-08-06
    Days: 252
    Episodes: 1
    Episode start range: [6092, 6092]

Sampling weights calculated:
  Training weights: [0.25010788 0.25010788 0.2498921  0.2498921 ]
  Validation weights: [0.24956822 0.24956822 0.25043178 0.25043178]
  Test weights: [1.]

============================================================
TIME SERIES SPLITTING COMPLETE
============================================================
Training blocks: 4
Validation blocks: 4
Test blocks: 1
Total training episodes: 16
Total validation episodes: 4
Total test episodes: 1
Actual validation ratio: 20.0%
============================================================