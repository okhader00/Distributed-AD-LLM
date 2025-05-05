import argparse
import flwr as fl
import torch
from tqdm import tqdm
from flwr.client import NumPyClient

from utils import TextDataset, evaluate
from prompt.ad_1_prompt import generate_prompt_setting_1
from prompt.ad_2_prompt import generate_prompt_setting_2
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

#Set to 1 for "Normal only" and set to 2 for "Normal + Anomaly"
ad_setting = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def call_model(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    output = model.generate(**inputs, max_new_tokens=256)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    
    #Extract JSON if present
    try:
        start = decoded.find("{")
        end = decoded.find("}") + 1
        parsed = json.loads(decoded[start:end])
        return parsed.get("anomaly_score", 0.0)
    except Exception:
        return 0.0


class InferenceClient(NumPyClient):
    def __init__(self, client_id, num_clients):
        self.client_id = client_id
        self.num_clients = num_clients

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Models:
        #mistralai/Mistral-7B-Instruct-v0.2
        #HuggingFaceH4/zephyr-7b-beta
        #TinyLlama/TinyLlama-1.1B-Chat-v1.0
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=True
        )

        self.dataset = TextDataset(data_name="bbc", model_name="llama")
        print(f"[Client {client_id}] Loaded {len(self.dataset)} samples.")
        print(f"[Client {self.client_id}] Model loaded to: {self.model.device}")

    def get_parameters(self, config): return []

    def fit(self, parameters, config): return [], 0, {}

    def evaluate(self, parameters, config):
        y_true = []
        y_score = []

        labels = self.dataset.get_labels()

        for i in tqdm(range(len(self.dataset)), desc=f"[Client {self.client_id}] Inference"):
            text = self.dataset[i]
            label = labels[i]
            if ad_setting == 1:
                prompt = generate_prompt_setting_1(
                    text=text,
                    normal_label_list=self.dataset.get_normal_label_list(),                   
                    normal_desc_dict=self.dataset.get_normal_desc_dict(),
                )
            elif ad_setting == 2:
                prompt = generate_prompt_setting_2(
                    text=text,
                    normal_label_list=self.dataset.get_normal_label_list(),
                    anomaly_label_list=self.dataset.get_anomaly_label_list(),
                    normal_desc_dict=self.dataset.get_normal_desc_dict(),
                    anomaly_desc_dict=self.dataset.get_anomaly_desc_dict(),
                )            

            score = call_model(self.model, self.tokenizer, prompt)
            y_true.append(label)
            y_score.append(score)

        return 0.0, len(y_true), {"y_true_json": json.dumps(y_true), "y_score_json": json.dumps(y_score)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("client_id", type=int)
    parser.add_argument("num_clients", type=int)
    parser.add_argument("--server-address", type=str, default="localhost:8080")
    args = parser.parse_args()

    client = InferenceClient(args.client_id, args.num_clients)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
