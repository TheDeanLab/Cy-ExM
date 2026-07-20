# Runtime guidance

- Keep runtime imports lightweight: never import ANTs, NumPy, SciPy, registration,
  or I/O modules here.
- Native-thread environment changes must happen only through an explicit callable;
  importing `cyexm.runtime` must not mutate `os.environ`.
- Resolve each native runtime independently. Explicit values beat inherited values,
  inherited values beat the default of one, and late changes are safe only when
  every effective value is exactly idempotent.
- Every local or distributed worker process must resolve and apply its explicit native
  thread configuration before importing ANTs, NumPy, or SciPy. This package reports
  thread policy; it must not initialize or own a scheduler.
