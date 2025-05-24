import torch

def pad_and_stack(features, pad_value=0.0):
    """
    Pads a list of tensors to the same maximum length and stacks them into a single tensor.

    Parameters:
    - features: List of tensors with different lengths [(L1, D), (L2, D), ...]
    - pad_value: Value to pad the tensors with, default is 0.0

    Returns:
    - A tensor of shape (batch_size, max_length, D)
    """
    # Determine the maximum sequence length
    max_length = max(f.shape[0] for f in features)
    embedding_dim = features[0].shape[-1]

    # Initialize a padded tensor with the pad value

    padded_features = []
    for i, feature in enumerate(features):
        length = feature.shape[0]

        if length != max_length:
            padding = torch.zeros(max_length - length, 1, embedding_dim).to(features[0].device)
            padded_features.append(torch.cat((feature, padding), dim=0))
        else:
            padded_features.append(feature)


    return torch.stack(padded_features)