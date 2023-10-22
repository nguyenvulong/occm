import pandas as pd
import argparse
from evaluate_metrics import compute_eer


def calculate_EER(eval_protocol_file, score_file):
    """
    Step:
        - load protocol file
        - load score file
        - calculate EER
    """
    pro_columns = ["sid", "utt","phy", "attack", "label"]
    # "./database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.eval.trl.txt"
    eval_protocol_file = pd.read_csv(eval_protocol_file, sep=" ", header=None)
    eval_protocol_file.columns = pro_columns
    # "./se_resnet34_eval_scores.txt"
    score_file = pd.read_csv(score_file, sep=" ", header=None)
    score_file.columns = ["utt", "score"]
    
    res = pd.merge(eval_protocol_file, score_file, on="utt")
    spoof_scores = res[res["label"] == "spoof"]["score"].values
    bonafide_scores = res[res["label"] == "bonafide"]["score"].values
    
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    print(f"EER = {eer*100.0}, threshold = {threshold}")

argparser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
argparser.add_argument('--eval_protocol_file', type=str, default="./database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.eval.trl.txt")
argparser.add_argument('--score_file', type=str, default="./se_resnet34_eval_scores.txt")
args = argparser.parse_args()

eval_protocol_file = args.eval_protocol_file
score_file = args.score_file

print(f"eval_protocol_file = {eval_protocol_file}")
print(f"score_file = {score_file}")

calculate_EER(eval_protocol_file, score_file)