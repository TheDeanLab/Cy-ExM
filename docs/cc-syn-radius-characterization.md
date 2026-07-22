# CC SyN radius characterization

:::{important}
This characterization is ongoing. It does not establish an optimized Cy-ExM
default or supersede the provisional scalar voxel radius of 4.
:::

Cy-ExM supports the `CC` nonlinear SyN metric with one scalar voxel radius.
The study compares the Mattes SyN baseline with CC radii 2, 4, and 6 or 8 on
approved cropped data. A radius spans the same number of voxels along each axis,
so anisotropic voxel spacing makes its physical neighborhood anisotropic.

A candidate is promoted only if it improves spatially resolved low-signal
residuals, preserves bright-region alignment and global similarity, produces no
folding or seam degradation, preserves Cy-ExM geometry and transform invariants,
and has acceptable runtime. Results will be added only after that evidence is
reviewed.
