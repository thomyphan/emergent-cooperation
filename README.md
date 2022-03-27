# Emergent Cooperation

## 1. Featured algorithms:

- Mutual Acknowledgment Token Exchange (MATE) [1]

## 2. Implemented domains

All available domains are listed in the table below. The labels are used for the commands below (in 5. and 6.).

| Domain   		| Label            | Description                                                       |
|---------------|------------------|-------------------------------------------------------------------|
| IPD           | `Matrix-IPD`     | Iterated Prisoner's Dilemma                 					   |
| ISH           | `Matrix-ISH`     | Iterated Stag Hunt Game                       					   |
| ISH           | `Matrix-ICG`     | Iterated Coordination Game                          			   |
| ISH           | `Matrix-IMP`     | Iterated Matching Pennies Game                					   |
| ISH           | `Matrix-IC`      | Iterated Chicken Game                         					   |
| Coin[2]       | `CoinGame-2`     | 2-player version of Coin                   					   |
| Coin[4]       | `CoinGame-4`     | 4-player version of Coin                   					   |
| Harvest[6]    | `Harvest-6`      | Harvest domain with 6 agents 				                       |
| Harvest[12]    | `Harvest-12`      | Harvest domain with 12 agents 				                       |

## 3. Implemented MARL algorithms

The reported MARL algorithms are listed in the tables below. The labels are used for the command below (in 5.).

| Algorithm       | Label                  |
|-----------------|------------------------|
| Random             | `Random`                |
| Naive Learner      | `IAC`                   |
| LOLA                | `LOLA`       |
| Gifting (Zero-Sum) | `Gifting-ZEROSUM`       |
| Gifting (Budget)   | `Gifting-BUDGET`       |
| LIO                | `LIO`       |
| MATE                | `MATE-TD`       |
| MATE (Defect=Complete)                | `MATE-TD-DEFECT_COMPLETE`       |
| MATE (Defect=Request)                | `MATE-TD-DEFECT_REQUEST`       |
| MATE (Defect=Response)                | `MATE-TD-DEFECT_RESPONSE`       |
| MATE (reward-based)     | `MATE-REWARD`       |

MATE, LIO, and Gifting can be trained with a communication failure rate of `X` with a value of `0.1`, `0.2`, `0.4`, or `0.8`:

| Algorithm       | Label                  |
|-----------------|------------------------|
 Gifting (Zero-Sum) | `Gifting-ZEROSUM-X`       |
| Gifting (Budget)   | `Gifting-BUDGET-X`       |
| LIO                | `LIO-X`       |
| MATE                | `MATE-TD-X`       |

## 4. Experiment parameters

The experiment parameters like the learning rate for training (`params["learning_rate"]`) or the number of episodes per epoch (`params["episodes_per_epoch"]`) are specified in `settings.py`. All other hyperparameters are set in the corresponding python modules in the package `mate/controllers`, where all final values as listed in the technical appendix are specified as default value.

All hyperparameters can be adjusted by setting their values via the `params` dictionary in `settings.py`.

## 5. Training

To train a MARL algorithm `M` (see tables in 3.) in domain `D` (see table in 2.), run the following command:

    python train.py D M

This command will create a folder with the name pattern `output/N-agents_domain-D_M_datetime` which contains the trained models (depending on the MARL algorithm).

`run.sh` is an example script for running all settings as specified in the paper.

## 6. Plotting

To generate learning plots for a particular domain `D` and evaluation mode `E` using metric `M` as presented in the paper, run the following command:

    python plot.py E D M

The command will load and display all the data of completed training runs that are stored in the folder which is specified in `params["output_folder"]` (see `settings.py`).

The evaluation mode `E` should be set to `True` when comparing `MATE` with other PI algorithms and `False` when comparing `MATE` with defective variants.

To generate communication robustness plots for a particular domain `D` using metric `M` as presented in the paper, run the following command:

    python plot_resilience.py E D M

`plot.sh` is an example script for plotting all experiments as presented in the paper.

**Note:** All plots will be output as .png, .pdf, and .svg in the folder `plots/`.

## 7. References

- [1] T. Phan et al., "Emergent Cooperation from Mutual Acknowledgment Exchange", in AAMAS 2022
