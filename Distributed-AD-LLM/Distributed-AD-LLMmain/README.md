# Distributed AD-LLM: Benchmarking Large Language Models for Anomaly Detection in a distributed setting

Using zero-shot anomaly detection with LLMs in a distributed setting using the flwr framework
Built from and inspired by: AD-LLM, the first benchmark that evaluates how large language models (LLMs) can assist with natural language processing (NLP) tasks in anomaly detection (AD).
[AD-LLM Repo](https://github.com/USC-FORTIS/AD-LLM)

## Environment Set-up
Use anaconda to create python environment and install required libraries:

```bash
# create the environment and activate it
conda create --name ad_llm python=3.11
conda activate ad_llm

# install basic packages
conda install numpy scipy scikit-learn matplotlib tqdm

# install PyTorch (adjust the CUDA version accordingly)
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# install PyOD
conda install -c conda-forge pyod

# install Flower
pip install flwr

# install libraries for models/huggingface
conda install -c conda-forge transformers
pip install --upgrade huggingface hub
pip install accelerate

## Environment Set-up
Use anaconda to create python environment and install required libraries:
```
## To run an experiment
Set desired `ad_setting`  in `client.py`. 1 for "Normal Only", 2 for "Normal + Anomaly".

```bash
# Start server
python fl_server.py
```

In seperate terminals depending on # of clients
```bash
python fl_client.py <current-client-id> <# of clients>
```
For example with 3 clients (each in seperate terminals)
```bash
python fl_client.py 0 3
```
```bash
python fl_client.py 1 3
```
```bash
python fl_client.py 2 3
```

## Notes
* We provide one example dataset "BBC News". Please check [NLP-ADBench](https://github.com/USC-FORTIS/NLP-ADBench) for more datasets (AG News, IMDB Reviews, N24 News, and SMS Spam) with the same setting.