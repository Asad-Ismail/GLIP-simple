import torch

def sigmoid_focal_loss(logits, targets, gamma, alpha):
      
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


def sigmoid_focal_loss_cuda(logits, targets, gamma, alpha):
    """
    GPU-optimized sigmoid focal loss that matches the behavior of the CPU implementation.
    """
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = logits.device

    # Class range tensor
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)

    # Targets and probabilities
    t = targets.unsqueeze(1)  # Shape: (N, 1)
    p = torch.sigmoid(logits)  # Shape: (N, num_classes)

    # Compute terms
    term1 = (1 - p) ** gamma * torch.log(p + 1e-6)  # Avoid log(0)
    term2 = p ** gamma * torch.log(1 - p + 1e-6)    # Avoid log(0)

    # Focal loss components
    pos_loss = -(t == class_range).float() * term1 * alpha
    neg_loss = -((t != class_range) & (t >= 0)).float() * term2 * (1 - alpha)

    return pos_loss + neg_loss



# Example inputs
logits = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 1.0, 2.5]], requires_grad=True).cuda()
targets = torch.tensor([1, 2]).cuda()  # Targets are 1-based for compatibility with your implementation

# Parameters
gamma = 2.0
alpha = 0.25

# Loss
loss_org = sigmoid_focal_loss(logits, targets,gamma, alpha)
loss_refactored = sigmoid_focal_loss_cuda(logits, targets,gamma, alpha)
print(loss_org,loss_refactored)

