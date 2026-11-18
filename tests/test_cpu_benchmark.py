import torch

from src.cpu_benchmark import benchmark_forward, probe_head_training
from src.models.architectures import build_mobilenet_v3_small


def test_cpu_forward_benchmark_preserves_modes_and_weights():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout()).train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = benchmark_forward(model, torch.ones(1, 3), warmup=1, repeats=3)
    assert len(result["measurements_ms"]) == 3
    assert 0 < result["min_ms"] <= result["median_ms"] <= result["p95_ms"]
    assert model.training and model[-1].training
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_training_probe_only_changes_disposable_copy():
    model = build_mobilenet_v3_small(pretrained=False).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = probe_head_training(model, batch_size=2, warmup=1, repeats=1)
    assert result["median_step_seconds"] > 0
    assert result["disposable_model_copy"]
    assert not model.training
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
