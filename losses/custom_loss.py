import torch
import torch.nn.functional as F

def compactness_loss(batch_embeddings):
    """
    Compactness loss.
    Calculates the Mahalanobis distance between each pair of samples and sums them up.
    Input:
        embeddings: tensor of shape (4, 128), where 4 is the number of samples and 128 is the embedding dimension
    Output:
        loss: scalar tensor representing the Mahalanobis distance loss
    """

    # Example: given batch of embeddings (4 samples with 128 dimensions)
    # batch_embeddings = torch.rand(4, 128)

    # Define the pairs you want to calculate the Mahalanobis distance for
    pairs = [(0, 1), (0, 2), (1, 2), (1, 3)]

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
        batch_embeddings: tensor of shape (4, 2), where "4" is the number of samples and "2" represents "bona fide" or "spoof" class
        labels: tensor of shape (4,), where each element is either 0 or 1 representing the label of the pair
  
    Output:
        loss: sum of 4 scalar tensors representing the total cross entropy loss
    """
    # Calculate the cross entropy loss for each pair of samples
    loss = F.cross_entropy(batch_embeddings, labels)

    # Sum up the losses
    total_loss = torch.sum(loss)
    return total_loss
