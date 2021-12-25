#!/bin/sh

echo "IPD:"
python plot.py True Matrix-IPD

echo "CoinGame-2:"
python plot.py True CoinGame-2
python plot.py True CoinGame-2 own_coin_prob

echo "CoinGame-4:"
python plot.py True CoinGame-4
python plot.py True CoinGame-4 own_coin_prob

echo "Harvest-6:"
python plot.py True Harvest-6
python plot.py True Harvest-6 equality
python plot.py True Harvest-6 sustainability
python plot.py True Harvest-6 peace

echo "Harvest-12:"
python plot.py True Harvest-12
python plot.py True Harvest-12 equality
python plot.py True Harvest-12 sustainability
python plot.py True Harvest-12 peace

echo "CoinGame-4 (defectors):"
python plot_resilience.py False CoinGame-4
python plot_resilience.py False CoinGame-4 own_coin_prob

echo "Harvest-12 (defectors):"
python plot_resilience.py False Harvest-12
python plot_resilience.py False Harvest-12 equality
python plot_resilience.py False Harvest-12 sustainability
python plot_resilience.py False Harvest-12 peace

echo "Resilience - CoinGame-4:"
python plot_resilience.py CoinGame-4
python plot_resilience.py CoinGame-4 own_coin_prob

echo "Resilience - Harvest-12:"
python plot_resilience.py Harvest-12
python plot_resilience.py Harvest-12 equality
python plot_resilience.py Harvest-12 sustainability
python plot_resilience.py Harvest-12 peace
