import numpy as np
import pytest
from scipy import ndimage

from cyexm.io.images import ants_to_external_array, external_to_ants_array
from cyexm.registration.transforms import (
    compose_affine_transforms,
    create_xy_rotation_transform,
)
from cyexm.spatial import ImageGeometry

pytestmark = pytest.mark.integration


def test_real_ants_seed_applies_documented_positive_physical_rotation() -> None:
    import ants

    moving_zyx = np.zeros((3, 11, 11), dtype=np.float32)
    moving_zyx[1, 5, 8] = 1.0
    moving = ants.from_numpy(external_to_ants_array(moving_zyx))
    geometry = ImageGeometry(
        shape=moving_zyx.shape,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        direction=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    transform = create_xy_rotation_transform(
        geometry=geometry,
        angle_deg=90.0,
    )

    transformed = transform.apply_to_image(
        moving,
        moving,
        interpolation="nearestNeighbor",
    )
    actual = ants_to_external_array(transformed.numpy())
    expected = ndimage.rotate(
        moving_zyx,
        # With identity direction, positive physical Y points down in a
        # conventional row-down display. Positive physical rotation therefore
        # appears clockwise and has the opposite SciPy display angle.
        angle=-90.0,
        axes=(1, 2),
        reshape=False,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    np.testing.assert_array_equal(actual, expected)


def test_real_ants_flattened_affine_matches_sequential_points() -> None:
    import ants

    translation = ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=3,
        matrix=np.eye(3),
        center=(0.0, 0.0, 0.0),
        translation=(3.0, -2.0, 1.0),
    )
    scale = ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=3,
        matrix=np.diag((1.2, 0.8, 1.0)),
        center=(4.0, 5.0, 6.0),
        translation=(0.0, 0.0, 0.0),
    )
    complete = compose_affine_transforms((translation, scale))
    point = (7.0, 9.0, 2.0)

    sequential = scale.apply_to_point(translation.apply_to_point(point))

    assert complete.apply_to_point(point) == pytest.approx(sequential, abs=1e-6)
