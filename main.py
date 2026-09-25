"""
Main Entry Point (revised)

Runs the complete pipeline end-to-end:
  1. preprocessing + feature engineering (two split protocols)
  2. K-Means clustering + elbow + PCA figures
  3. three classifiers evaluated on BOTH protocols, results saved to JSON

Usage:
  python main.py [--data PATH_TO_AirPollutionSeoul]
If --data is omitted, the AIR_DATA_PATH environment variable, or a local
./AirPollutionSeoul folder, is used.
"""

import argparse
import time

from data_preprocessing import get_preprocessed_data
from task_clustering import perform_clustering
from task_classification import perform_classification


def main():
    parser = argparse.ArgumentParser(description='Seoul Air Quality ML pipeline')
    parser.add_argument('--data', default=None,
                        help='Path to the AirPollutionSeoul data folder')
    parser.add_argument('--k', type=int, default=3, help='K for K-Means')
    args = parser.parse_args()

    t0 = time.time()
    data = get_preprocessed_data(base_path=args.data)

    # 2. Clustering on the training half of the primary (chronological) split.
    perform_clustering(data['chronological']['X_train'], optimal_k=args.k)

    # 3. Classification on both protocols.
    perform_classification({'chronological': data['chronological'],
                            'random-stratified': data['random-stratified']})

    print(f'\nPipeline finished in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
