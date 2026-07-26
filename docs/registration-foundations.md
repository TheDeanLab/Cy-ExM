# Registration foundations

Cy-ExM keeps scientific geometry and transform decisions explicit. Its accepted invariants state that public pixel sizes are micrometers in `(x, y, z)` order; NumPy volumes are `(z, y, x)`; and affine application is the first fixed-grid boundary. Native reference, registration-channel, and dependent-channel shapes can differ, but the scalar and mask outputs, nonlinear inputs and outputs, full warp, and transformed dependent channels use the fixed-reference geometry.

The same invariants state that masks use values greater than zero; affine precedes nonlinear registration; scalar data use linear interpolation and masks use nearest-neighbor interpolation; affine is TRSAA/Mattes with 32 bins; and nonlinear registration is SyNOnly. These choices are part of the registration contract rather than display-only defaults.

The nonlinear metric may be Mattes or `CC`. CC radius is one positive scalar voxel radius, and omitted CC radius uses provisional value 4. A CC radius is not a per-axis physical radius. On anisotropic data, equal voxel extents produce an anisotropic physical neighborhood, so radius selection requires characterization rather than an assumed universal optimum.

Finally, native ITK/BLAS-family threads default to one unless inherited or configured explicitly. Keep native-thread settings explicit in execution profiles when comparing runtimes or operating worker-backed executors.

## Optional background-subtracted asinh features

Schema version 1 permits an optional `registration.preprocessing` mapping. Omission
disables preprocessing. An empty mapping enables the default
`filter_sigma_zyx: [12, 12, 12]`; a positive scalar expands to all axes, and a
three-value form is interpreted in NumPy `(z, y, x)` voxel order. These sigma values
are voxel counts, not micrometers.

Cy-ExM crops raw images first, creates greater-than-zero masks from those raw values,
and then computes a feature independently for the fixed and moving registration
channels. It deterministically samples each signed Gaussian-background residual as
`residual.ravel()[::max(1, residual.size // 2_000_000)]` before clipping, and uses
its MAD to calculate `max(1.4826 × MAD(sample), float32 epsilon)`. The feature is
the positive portion of the full signed residual divided by that scale and passed
through `asinh`. The Gaussian boundary mode is reflective. TRSAA and nonlinear
registration see the float32 features and raw-derived masks.

Transforms remain the scientific result. The accepted affine and warp are applied to
the original raw registration channel; dependent channels are likewise transformed
from their original sources. Pre-, linear-, and nonlinear-registration metrics use
raw intensities. The public registered TIFF is clipped to uint16 without intensity
normalization. No normalized feature TIFF is a public artifact.
