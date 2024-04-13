import torch


class AngularMarginLoss(torch.nn.Module):
    def __init__(self,
                 embedding_size: int,
                 num_classes: int,
                 scale: float,
                 margin: float,
                 eps: float=1e-6):
        super().__init__()
        self.logic_linear = torch.nn.Linear(embedding_size, num_classes, bias=False)
        self.scale = scale
        self.margin = margin
        self.eps = eps

    def forward(self, embedding: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.logic_linear.weight.data = torch.nn.functional.normalize(self.logic_linear.weight.data)

        # # Normalize inputs
        # inputs_norms = torch.norm(embedding, p=2, dim=1)
        # normalized_inputs = embedding / inputs_norms.unsqueeze(-1).repeat(
        #     1, embedding.size(1)
        # )

        # Set scale
        scales = torch.tensor([self.scale], device=embedding.device).repeat(embedding.size(0))

        # Cosine similarity is given by a simple dot product,
        # given that we normalized both weights and inputs
        cosines = self.logic_linear(embedding).clamp(-1, 1)

        # Recover angles from cosines computed
        # from the previous linear layer
        angles = torch.arccos(cosines)

        # Compute loss numerator by converting angles back to cosines,
        # after adding penalties, as if they were the output of the
        # last linear layer
        numerator = scales.unsqueeze(-1) * (
                torch.cos(angles + self.margin)
        )
        numerator = torch.diagonal(numerator.transpose(0, 1)[targets])

        # Compute loss denominator
        excluded = torch.cat(
            [
                scales[i]
                * torch.cat((cosines[i, :y], cosines[i, y + 1:])).unsqueeze(0)
                for i, y in enumerate(targets)
            ],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(excluded), dim=1)

        # Compute cross-entropy loss
        loss = -torch.mean(numerator - torch.log(denominator + self.eps))

        return loss
