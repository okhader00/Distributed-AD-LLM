import flwr as fl
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score
from flwr.common import Parameters, ndarrays_to_parameters

class InferenceAggregator(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.logs = []

    def initialize_parameters(self, client_manager):
        dummy = ndarrays_to_parameters([np.array([])])
        return super().initialize_parameters(client_manager)
    
    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, rnd, results, failures):
        all_y_true = []
        all_y_score = []

        for _, eval_res in results:
            y_true = json.loads(eval_res.metrics["y_true_json"])
            y_score = json.loads(eval_res.metrics["y_score_json"])
            all_y_true.extend(y_true)
            all_y_score.extend(y_score)

        y_pred = [1 if s > 0.5 else 0 for s in all_y_score]

        # Compute metrics
        auc = roc_auc_score(all_y_true, all_y_score)
        auprc = average_precision_score(all_y_true, all_y_score)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_y_true, [int(s > 0.5) for s in all_y_score], average="binary", zero_division=0
        )
        accuracy = accuracy_score(all_y_true, y_pred)

        metrics = {
            "round": rnd,
            "roc_auc": auc,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

        print(f"\n Round {rnd} Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        self.logs.append(metrics)
        pd.DataFrame(self.logs).to_csv("federated_anomaly_results.csv", index=False)

        return 0.0, metrics

def start_server():
    strategy = InferenceAggregator()
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy
    )

if __name__ == "__main__":
    print("Federated Inference Server started")
    start_server()
