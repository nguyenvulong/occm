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

    # Get the embeddings of the pairs
    batch_embeddings = torch.stack([batch_embeddings[i] for i, j in pairs])
    print(f"batch_embeddings = {batch_embeddings.shape}")
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

def triplet_loss(batch_embeddings, margin=0.2):
    """Calculate triplet loss using Euclidean distance.
    Expects batch_embeddings to be ordered as [bona1, bona2, spoof1].
    Uses a default margin of 0.2.

    Args:
        batch_embeddings (tensor): embeddings of bona1, bona2, and spoof1.
                                   Expected shape [3, embedding_dim].
        margin (float, optional): Margin by which the distance between the
                                  negative and positive should be greater.
                                  Defaults to 0.2.

    Returns:
        torch.Tensor: Triplet loss.
    """
    # Calculate pairwise distances
    bona2bona = F.pairwise_distance(batch_embeddings[0].unsqueeze(0), 
                                    batch_embeddings[1].unsqueeze(0), p=2)
    bona2spoof = F.pairwise_distance(batch_embeddings[0].unsqueeze(0), 
                                     batch_embeddings[2].unsqueeze(0), p=2)
    
    # Calculate triplet loss with margin
    # Ensure the loss is non-negative
    loss = F.relu(bona2bona - bona2spoof + margin)  
    # .mean()  # If multiple triplets, return the mean loss.
    return loss

def euclidean_distance_loss(batch_embeddings):
    # Initialize the loss to 0
    loss = 0.0
    pairs = [(0, 1), (0, 2), (0, 3), (2, 1), (2, 3)]

    # Calculate the Euclidean distance for each pair and add to the loss
    for i, j in pairs:
        # Compute the L2 distance between the two embeddings
        distance = F.pairwise_distance(batch_embeddings[i].unsqueeze(0), batch_embeddings[j].unsqueeze(0), p=2)
        # Add the distance to the total loss
        loss += distance
    
    # Optionally, you can average the loss over the number of pairs if needed
    loss = loss / len(pairs)

    return loss



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
    weight = 2
    # Calculate the cross entropy loss for each pair of samples
    
    # loss = torch.sum(F.cross_entropy(batch_embeddings, labels, reduction='none'))
    # calculate the cross entropy loss of the first sample
    loss_1 = F.cross_entropy(batch_embeddings[0].unsqueeze(0), labels[0].unsqueeze(0), reduction='none')
    loss_2 = F.cross_entropy(batch_embeddings[4].unsqueeze(0), labels[4].unsqueeze(0), reduction='none')
    return (weight * loss_1 + loss_2)/3
