import torch
import torch.nn.functional as F

def compactness_loss(batch_embeddings):
    """
    Compactness loss.
    Calculates the Mahalanobis distance between each pair of samples and sums them up.
    Input:
        embeddings: tensor of shape (8, 128), where 8 is the number of samples and 128 is the embedding dimension
    Process:
        only use the first 4 samples (0, 1, 2, 3) to calculate the Mahalanobis distance
        they are bona1, bona2, bona3 and bona4
        calculate the Mahalanobis distance between the following pairs
            bona1 and bona2
            bona1 and bona3
            bona1 and bona4
            bona3 and bona2
            bona3 and bona4
            
    Output:
        loss: scalar tensor representing the Mahalanobis distance loss
    """

    # Example: given batch of embeddings (4 samples with 128 dimensions)
    # batch_embeddings = torch.rand(4, 128)

    # Define the pairs you want to calculate the Mahalanobis distance for
    pairs = [(0, 1), (0, 2), (0, 3), (2, 1), (2, 3)] 

    # Compute the sample mean
    mean_embedding = torch.mean(batch_embeddings, dim=0)

    # Compute the sample covariance matrix
    cov_matrix = torch.mm((batch_embeddings - mean_embedding).t(), batch_embeddings - mean_embedding) / (batch_embeddings.size(0) - 1)

    # Calculate the Mahalanobis distance for each pair
    mahalanobis_distances = []
    for i, j in pairs:
        diff = batch_embeddings[i] - batch_embeddings[j]
        mahalanobis_distance = torch.mm(torch.mm(diff.unsqueeze(0), torch.inverse(cov_matrix)), diff.unsqueeze(1))
        mahalanobis_distances.append(mahalanobis_distance)

    # Sum up the Mahalanobis distances
    total_mahalanobis_distance = torch.sum(torch.stack(mahalanobis_distances))

    return total_mahalanobis_distance


def descriptiveness_loss(batch_embeddings, labels):
    """
    Descriptiveness loss.
    Calculates the sum of cross entropy loss of 4 pairs of samples.
    
    Inputs:
        batch_embeddings: tensor of shape (8, 2), where "8" is the number of samples and "2" represents "bona fide" or "spoof" class
                          the samples are: bona1, bona2, bona3, bona4, spoof1, spoof2, spoof3, spoof4
        labels: tensor of shape (8,), where each element is either 0 or 1 representing the label of the sample
  
    Process:
        calculate the cross entropy loss of 8 samples with their corresponding labels
    Output:
        loss: sum the cross entropy loss of all 8 samples
    """
    # Calculate the cross entropy loss for each pair of samples
    
    loss = torch.sum(F.cross_entropy(batch_embeddings, labels, reduction='none'))
    return loss
