//                             Meles BPs Example
//
// This example illustrates a simple usage of libCEED and PETSc to create
// matrix-free operators for the CEED Benchmark Problems.

use meles::prelude::*;
use petsc_rs::prelude::*;

// ----------------------------------------------------------------------------
// BPs
// ----------------------------------------------------------------------------
fn main() -> meles::Result<()> {
    let petsc = petsc_rs::Petsc::init_no_args()?;
    let mut meles = meles::Meles::new(&petsc, "./bps1.yml", meles::MethodType::BenchmarkProblem)?;
    Ok(())
}

// ----------------------------------------------------------------------------
