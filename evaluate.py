import argparse
from sklearn.metrics import confusion_matrix

def eval_dict(file_path):
    """_summary_

    Given a metadata file for DF eval
                eval-package/keys/DF/CM/trial_metadata.txt
                Example
                LA_0043 DF_E_2000026 mp3m4a asvspoof A09 spoof notrim eval traditional_vocoder - - - -

    Create a dictionary with key = file name, value = label. Where key is the second column and value is the sixth column.
        
    """
    eval_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            file_name = line.split(" ")[1]
            label = line.split(" ")[5]
            eval_dict[file_name] = label
    return eval_dict

def load_proto_file(file_path):
    """load the protocol file which contains the file names only
       and return a list of file names

    Args:
        file_path (str): path to the protocol file
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        file_list = []
        for line in lines:
            line = line.strip()
            file_list.append(line)
    return file_list

def load_score_file(file_path):
    """load the score file and return the score list
       example:
            0.02207140438258648, 0 
            0.01588536612689495, 1

    Args:
        file_path (str): path to the score file
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        score_list = []
        for line in lines:
            line = line.strip()
            score = float(line.split(",")[0])
            score_list.append(score)
    return score_list

if __name__=="__main__":
    
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_file", type=str, default="score.txt")
    parser.add_argument("--protocol_file", type=str, default="protocol.txt")
    parser.add_argument("--metadata_file", type=str, default="metadata.txt")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    
    # load the protocol file, the score file, and the metadata file
    proto = load_proto_file(args.protocol_file)
    scores = load_score_file(args.score_file)
    metadata = eval_dict(args.metadata_file)
    
    # for each file in the protocol file, get the score and the label
    # compare the score with the threshold
    # if the score is greater than the threshold, the prediction is spoof
    # and bonafide otherwise
    
    # create two lists: one for the labels and one for the predictions
    labels = []
    predictions = []
    for file_name in proto:
        score = scores[proto.index(file_name)]
        label = metadata[file_name]
        labels.append(label)
        if score > args.threshold:
            predictions.append("spoof")
        else:
            predictions.append("bonafide")

    # number of bona fide and spoof in labels
    bona_fide = labels.count("bonafide")
    spoof = labels.count("spoof")
    print(f"bona fide = {bona_fide}")
    print(f"spoof = {spoof}")

    # calculate the confusion matrix
    cm = confusion_matrix(labels, predictions)
    print(cm)
    # print TP, TN, FP, FN
    print(f"TP = {cm[0][0]}")
    print(f"TN = {cm[1][1]}")
    print(f"FP = {cm[0][1]}")
    print(f"FN = {cm[1][0]}")
    