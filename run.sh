#!/bin/sh

# 1. Performance Evaluation

python train.py Matrix-IPD MATE-REWARD
python train.py Matrix-IPD MATE-TD
python train.py Matrix-IPD Gifting-ZEROSUM
python train.py Matrix-IPD Gifting-BUDGET
python train.py Matrix-IPD IAC

python train.py CoinGame-2 MATE-REWARD
python train.py CoinGame-2 MATE-TD
python train.py CoinGame-2 LIO
python train.py CoinGame-2 Gifting-ZEROSUM
python train.py CoinGame-2 Gifting-BUDGET
python train.py CoinGame-2 IAC

python train.py CoinGame-4 MATE-REWARD
python train.py CoinGame-4 MATE-TD
python train.py CoinGame-4 LIO
python train.py CoinGame-4 Gifting-ZEROSUM
python train.py CoinGame-4 Gifting-BUDGET
python train.py CoinGame-4 IAC

python train.py Harvest-6 MATE-REWARD
python train.py Harvest-6 MATE-TD
python train.py Harvest-6 LIO
python train.py Harvest-6 Gifting-ZEROSUM
python train.py Harvest-6 Gifting-BUDGET
python train.py Harvest-6 IAC

python train.py Harvest-12 MATE-REWARD
python train.py Harvest-12 MATE-TD
python train.py Harvest-12 LIO
python train.py Harvest-12 Gifting-ZEROSUM
python train.py Harvest-12 Gifting-BUDGET
python train.py Harvest-12 IAC

# 2. Protocol Defectors

python train.py CoinGame-4 MATE-TD-DEFECT_COMPLETE
python train.py CoinGame-4 MATE-TD-DEFECT_REQUEST
python train.py CoinGame-4 MATE-TD-DEFECT_RESPONSE

python train.py Harvest-12 MATE-TD-DEFECT_COMPLETE
python train.py Harvest-12 MATE-TD-DEFECT_REQUEST
python train.py Harvest-12 MATE-TD-DEFECT_RESPONSE

# 3. Communication Failures

python train.py CoinGame-4 MATE-TD-0.1
python train.py CoinGame-4 MATE-TD-0.2
python train.py CoinGame-4 MATE-TD-0.4
python train.py CoinGame-4 MATE-TD-0.8
python train.py CoinGame-4 LIO-0.1
python train.py CoinGame-4 LIO-0.2
python train.py CoinGame-4 LIO-0.4
python train.py CoinGame-4 LIO-0.8

python train.py Harvest-12 MATE-TD-0.1
python train.py Harvest-12 MATE-TD-0.2
python train.py Harvest-12 MATE-TD-0.4
python train.py Harvest-12 MATE-TD-0.8
python train.py Harvest-12 LIO-0.1
python train.py Harvest-12 LIO-0.2
python train.py Harvest-12 LIO-0.4
python train.py Harvest-12 LIO-0.8
