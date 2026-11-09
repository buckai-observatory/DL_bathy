"""
Unit tests for inference.py -- the public inference API (issue #23).

No network access or real checkpoint is needed: models are built with
encoder_weights=None or replaced by a tiny in-memory model, and
huggingface_hub.hf_hub_download is monkeypatched to return a local file.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
from torch import nn

import inference


class TinyModel(nn.Module):
    """A 1x1 convolution mapping 12 bands to 1 output, like the real model."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(12, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return TinyModel().eval()


# ---------------------------------------------------------------------------
# _resolve_encoder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias, expected", [
    ("convnext",            "tu-convnext_large"),
    ("convnext_large",      "tu-convnext_large"),
    ("tu-convnext_large",   "tu-convnext_large"),
    ("efficientnet-b4",     "efficientnet-b4"),
    ("resnet50",            "resnet50"),
    ("deeplabv3_resnet50",  "resnet50"),
    ("resnet101",           "resnet101"),
    ("deeplabv3_resnet101", "resnet101"),
    ("ResNet50",            "resnet50"),   # case-insensitive
])
def test_resolve_encoder_aliases(alias, expected):
    assert inference._resolve_encoder(alias) == expected


def test_resolve_encoder_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown encoder_name"):
        inference._resolve_encoder("vgg16")


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def test_build_model_matches_training_architecture():
    """build_model must reproduce dl_bathy.train.load_model's network exactly,
    so that checkpoints written by training load with strict=True."""
    from dl_bathy.train import load_model as train_load_model

    model = inference.build_model("resnet50")
    assert not model.training
    assert model.encoder._in_channels == 12
    assert model.encoder._output_stride == 16

    with torch.no_grad():
        y = model(torch.zeros(1, 12, 64, 64))
    assert y.shape == (1, 1, 64, 64)

    trained, *_ = train_load_model("resnet50", torch.device("cpu"), encoder_weights=None)
    expected = {k: v.shape for k, v in trained.state_dict().items()}
    assert {k: v.shape for k, v in model.state_dict().items()} == expected


def test_build_model_rejects_unknown_encoder():
    with pytest.raises(ValueError):
        inference.build_model("vgg16")


# ---------------------------------------------------------------------------
# predict_depth
# ---------------------------------------------------------------------------

def test_predict_depth_numpy_and_torch_inputs_agree():
    model = _tiny_model()
    rng = np.random.default_rng(0)
    bands = rng.uniform(0, 3000, (12, 8, 10))            # raw L2A digital numbers

    out_np = inference.predict_depth(model, bands)
    out_t = inference.predict_depth(model, torch.from_numpy(bands))

    assert isinstance(out_np, np.ndarray)
    assert out_np.shape == (8, 10)
    assert out_np.dtype == np.float32
    np.testing.assert_allclose(out_np, out_t)

    # Issue #30: the model must see log(band/10000 + 1e-6), as in training
    normalised = np.log(bands / 10000 + 1e-6).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(normalised)[None])[0, 0].numpy()
    np.testing.assert_allclose(out_np, expected, rtol=1e-5)
    np.testing.assert_allclose(inference.predict_depth(model, normalised, normalize=False), expected, rtol=1e-5)


def test_predict_depth_input_matches_sdb_split_patches(scene):
    # A raw scene window, normalised by predict_depth, must equal the bands
    # sdb split wrote for the same patch -- the values the model trained on.
    import rasterio
    from rasterio.windows import from_bounds
    from dl_bathy.bands import normalize_reflectance
    from dl_bathy.cli import main
    from dl_bathy.patch_io import read_patch_tif
    main(["split", "--config", str(scene)])
    patch = next((scene.parent / "Data" / "train").glob("*.tif"))
    X, _, _, _ = read_patch_tif(str(patch))
    with rasterio.open(patch) as p:
        bounds = p.bounds
    with rasterio.open(next((scene.parent / "s2").glob("*.tif"))) as src:
        raw = src.read(list(range(1, 13)), window=from_bounds(*bounds, transform=src.transform)).astype(float)
    assert raw.shape == X.shape
    np.testing.assert_allclose(normalize_reflectance(raw), X, rtol=1e-5, atol=1e-5)

    model = _tiny_model()
    np.testing.assert_allclose(inference.predict_depth(model, raw),
                               inference.predict_depth(model, X, normalize=False), rtol=1e-4, atol=1e-5)


def test_predict_depth_returns_nan_where_input_is_invalid():
    bands = np.full((12, 4, 4), 1000.0)
    bands[3, 1, 2] = np.nan
    out = inference.predict_depth(_tiny_model(), bands)
    assert np.isnan(out[1, 2]) and np.isfinite(np.delete(out.ravel(), 1 * 4 + 2)).all()


@pytest.mark.parametrize("shape", [(11, 8, 8), (13, 8, 8), (12, 8), (1, 12, 8, 8)])
def test_predict_depth_rejects_wrong_shape(shape):
    with pytest.raises(ValueError, match=r"\(12, H, W\)"):
        inference.predict_depth(_tiny_model(), np.zeros(shape, np.float32))


# ---------------------------------------------------------------------------
# load_model (checkpoint handling, no download)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Make load_model build a TinyModel and 'download' a local checkpoint.

    Returns a function that saves *obj* as the checkpoint and records the
    arguments hf_hub_download was called with in ``calls``.
    """
    import huggingface_hub

    ckpt = tmp_path / "deeplabv3_bathy_final.pth"
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(ckpt)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(inference, "build_model",
                        lambda encoder_name, device=None: _tiny_model(seed=1).to(device))

    def save(obj):
        torch.save(obj, ckpt)
        return calls

    return save


def _assert_same_weights(model, reference):
    ref = reference.state_dict()
    for k, v in model.state_dict().items():
        assert torch.equal(v, ref[k]), k


@pytest.mark.parametrize("wrap", ["bare", "bundle", "module_prefix", "bundle_module_prefix"])
def test_load_model_state_dict_formats(fake_hub, wrap):
    reference = _tiny_model(seed=0)
    state = reference.state_dict()
    if "module_prefix" in wrap:
        state = {"module." + k: v for k, v in state.items()}
    if "bundle" in wrap:
        state = {"state_dict": state, "epoch": 3}
    fake_hub(state)

    model = inference.load_model(device="cpu")

    assert not model.training
    _assert_same_weights(model, reference)


def test_load_model_passes_hub_arguments(fake_hub):
    calls = fake_hub(_tiny_model().state_dict())

    inference.load_model(repo_id="org/repo", filename="w.pth", revision="v1",
                         cache_dir="cache", device="cpu")

    assert calls == [dict(repo_id="org/repo", filename="w.pth", revision="v1", cache_dir="cache")]


def test_load_model_defaults_to_published_weights(fake_hub):
    calls = fake_hub(_tiny_model().state_dict())

    inference.load_model(device="cpu")

    assert calls[0]["repo_id"] == inference.HF_DEFAULT_REPO_ID
    assert calls[0]["filename"] == inference.HF_DEFAULT_FILENAME
    assert calls[0]["revision"] is None


def test_load_model_rejects_mismatched_checkpoint(fake_hub):
    """strict=True loading must fail loudly on a checkpoint for another architecture."""
    fake_hub({"other.weight": torch.zeros(1)})
    with pytest.raises(RuntimeError):
        inference.load_model(device="cpu")
