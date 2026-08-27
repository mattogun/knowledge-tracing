import torch

from kt.models.dkt import DKT


def test_forward_shape():
    model = DKT(n_items=100, d=8, hidden=16)
    b, length = 3, 10
    items = torch.randint(1, 101, (b, length))
    resps = torch.randint(1, 3, (b, length))
    nxt = torch.randint(1, 101, (b, length))
    assert model(items, resps, nxt).shape == (b, length)


def test_overfits_tiny_batch():
    # The classic sanity check: if it cannot memorise 4 sequences, it is broken.
    torch.manual_seed(0)
    model = DKT(n_items=50, d=16, hidden=32)
    items = torch.randint(1, 51, (4, 20))
    resps = torch.randint(1, 3, (4, 20))
    nxt = torch.randint(1, 51, (4, 20))
    labels = torch.randint(0, 2, (4, 20)).float()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()
    loss = None
    for _ in range(300):
        opt.zero_grad()
        loss = lossf(model(items, resps, nxt), labels)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
