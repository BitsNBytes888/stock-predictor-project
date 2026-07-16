import numpy as np
import pytest

from backend.ml.preprocessing.scaling import StandardScaler3D


@pytest.fixture
def X_3d():
    rng = np.random.default_rng(7)
    return rng.normal(loc=5.0, scale=2.0, size=(40, 10, 3)).astype(np.float64)


def test_fit_sets_mean_and_std_shapes(X_3d):
    scaler = StandardScaler3D().fit(X_3d)
    assert scaler.mean_.shape == (3,)
    assert scaler.std_.shape == (3,)


def test_transform_zero_centers(X_3d):
    scaler = StandardScaler3D().fit(X_3d)
    X_t = scaler.transform(X_3d)
    flat = X_t.reshape(-1, X_t.shape[-1])
    assert np.abs(flat.mean(axis=0)).max() < 1e-10


def test_transform_unit_scales(X_3d):
    scaler = StandardScaler3D().fit(X_3d)
    X_t = scaler.transform(X_3d)
    flat = X_t.reshape(-1, X_t.shape[-1])
    assert np.abs(flat.std(axis=0) - 1.0).max() < 1e-10


def test_zero_std_column_no_divide_by_zero():
    X = np.ones((20, 5, 2))
    X[:, :, 1] = np.random.randn(20, 5)
    scaler = StandardScaler3D().fit(X)
    assert scaler.std_[0] == 1.0
    X_t = scaler.transform(X)
    assert np.isfinite(X_t).all()
    assert (X_t[:, :, 0] == 0.0).all()


def test_fit_transform_equals_fit_then_transform(X_3d):
    s1 = StandardScaler3D()
    out1 = s1.fit_transform(X_3d)

    s2 = StandardScaler3D()
    s2.fit(X_3d)
    out2 = s2.transform(X_3d)

    np.testing.assert_array_almost_equal(out1, out2)


def test_transform_uses_fit_statistics(X_3d):
    scaler = StandardScaler3D().fit(X_3d)
    rng = np.random.default_rng(99)
    X_test = rng.normal(loc=10.0, scale=3.0, size=(5, 10, 3))
    X_t = scaler.transform(X_test)
    expected = (X_test - scaler.mean_) / scaler.std_
    np.testing.assert_array_almost_equal(X_t, expected)
