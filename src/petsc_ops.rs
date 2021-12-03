use crate::prelude::*;

// -----------------------------------------------------------------------------
// Apply the local action of a libCEED operator and store result in PETSc vector
//   i.e. compute A X = Y
// -----------------------------------------------------------------------------
pub(crate) fn apply_local_ceed_op<'a>(
    x: &petsc_rs::vector::Vector<'a>,
    y: &mut petsc_rs::vector::Vector<'a>,
    meles: &&Meles,
) -> petsc_rs::Result<()> {
    Ok(())
}

// -----------------------------------------------------------------------------
// Compute the diagonal of an operator via libCEED
// -----------------------------------------------------------------------------
pub(crate) fn get_diagonal_ceed<'a>(
    d: &mut petsc_rs::vector::Vector<'a>,
    meles: &&Meles,
) -> petsc_rs::Result<()> {
    Ok(())
}
// -----------------------------------------------------------------------------
