# Registration foundations

Cy-ExM keeps scientific geometry and transform decisions explicit. Its accepted invariants state that public pixel sizes are micrometers in `(x, y, z)` order; NumPy volumes are `(z, y, x)`; and affine application is the first fixed-grid boundary. Native reference, registration-channel, and dependent-channel shapes can differ, but the scalar and mask outputs, nonlinear inputs and outputs, full warp, and transformed dependent channels use the fixed-reference geometry.

The same invariants state that masks use values greater than zero; affine precedes nonlinear registration; scalar data use linear interpolation and masks use nearest-neighbor interpolation; affine is TRSAA/Mattes with 32 bins; and nonlinear registration is SyNOnly. These choices are part of the registration contract rather than display-only defaults.

The nonlinear metric may be Mattes or `CC`. CC radius is one positive scalar voxel radius, and omitted CC radius uses provisional value 4. A CC radius is not a per-axis physical radius. On anisotropic data, equal voxel extents produce an anisotropic physical neighborhood, so radius selection requires characterization rather than an assumed universal optimum.

Finally, native ITK/BLAS-family threads default to one unless inherited or configured explicitly. Keep native-thread settings explicit in execution profiles when comparing runtimes or operating worker-backed executors.
