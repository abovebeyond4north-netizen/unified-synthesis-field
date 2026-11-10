import Mathlib

set_option autoImplicit false

namespace USF

/-- A non-novel verification smoke test. Not evidence of autonomous discovery. -/
theorem ring_example (x : ℤ) : (x + 1) ^ 2 = x ^ 2 + 2 * x + 1 := by
  ring

end USF

#print axioms USF.ring_example
