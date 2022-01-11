//                             Meles BPs Example
//
// This example illustrates a simple usage of libCEED to compute the volume of a
// 3D body using matrix-free application of a mass operator.  Arbitrary mesh and
// solution orders in 1D, 2D and 3D are supported from the same code.
//
// The example has no dependencies, and is designed to be self-contained. For
// additional examples that use external discretization libraries (MFEM, PETSc,
// etc.) see the subdirectories in libceed/examples.
//
// All libCEED objects use a Ceed device object constructed based on a command
// line argument (-ceed).

use meles::prelude::*;
use petsc_rs::prelude::*;

// ----------------------------------------------------------------------------
// BPs
// ----------------------------------------------------------------------------
fn main() -> meles::Result<()> {
    let petsc = petsc_rs::Petsc::init_no_args()?;
    let mut meles = meles::Meles::new(&petsc, "./bps1.yml")?;
    let dm = meles.dm(meles::MethodType::BenchmarkProblem)?;
    Ok(())
}

// ----------------------------------------------------------------------------
