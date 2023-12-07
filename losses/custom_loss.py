import torch
import torch.nn.functional as F

def compactness_loss(batch_embeddings):
    """Calculate the Euclidean distance between a bona fide and the mean
    of the rest of the bona fides in the batch, then calculate the average
    of these distances. Batch has a shape of `torch.Size([12, 128])`

    Args:
        batch_embeddings (tensor): embeddings of bona fide batch.
                                   Expected shape [batch_size, embedding_dim].
    """
    distances = []
    # Only 6 bona fides in the batch, so iterate through them.
    batch_embeddings = batch_embeddings[:6]
    for i in range(len(batch_embeddings)):
        bona_fide = batch_embeddings[i]
        # Exclude the i-th embedding to calculate the mean of the others.
        others_mean = torch.mean(torch.cat((batch_embeddings[:i], batch_embeddings[i+1:]), dim=0), dim=0)
        
        # Expand dimensions to match for pairwise_distance calculation.
        bona_fide = bona_fide.unsqueeze(0)
        others_mean = others_mean.unsqueeze(0)
        # Calculate and store the distance.
        distance = F.pairwise_distance(bona_fide, others_mean, p=2)
        distances.append(distance)
        
    # Convert the list of distances into a tensor and compute the mean.
    return torch.mean(torch.cat(distances))


def triplet_loss(batch_embeddings, margin=9.0):
    """Calculate triplet loss using Euclidean distance.
    Expects batch_embeddings to be ordered as [bona1, bona2, spoof1].
    Uses a default margin of 0.2.

    Args:
        batch_embeddings (tensor): embeddings of bona1, bona2, and spoof1.
                                   Expected shape [3, embedding_dim].
        margin (float, optional): Margin by which the distance between the
                                  negative and positive should be greater.
                                  Defaults to 9.0.

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
        calculate the cross entropy loss of 12 samples with their corresponding labels
    Output:
        loss: sum the cross entropy loss of all 12 samples
    """
    # Calculate the cross entropy loss for each pair of samples
    
    loss = torch.sum(F.cross_entropy(batch_embeddings, labels, reduction='none'))
    # calculate the cross entropy loss of the first sample
    # loss_1 = F.cross_entropy(batch_embeddings[0].unsqueeze(0), labels[0].unsqueeze(0), reduction='none')
    # loss_2 = F.cross_entropy(batch_embeddings[4].unsqueeze(0), labels[4].unsqueeze(0), reduction='none')
    return loss / len(batch_embeddings)
