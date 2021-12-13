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
    // Global to local
    meles
        .mesh_dm
        .global_to_local(&x, petsc_rs::InsertMode::INSERT_VALUES, &mut meles.x_loc)?;
    // Apply libCEED operator
    {
        let x_loc_view = meles.x_loc.view()?;
        let x_loc_borrow = meles.x_loc_ceed.set_borrowed_slice(&mut x_loc_view)?;
        let mut y_loc_view = meles.y_loc.view_mut()?;
        let y_loc_borrow = meles.y_loc_ceed.set_borrowed_slice(&mut y_loc_view)?;

        meles.op.apply(&meles.x_loc_ceed, &mut meles.y_loc_ceed)?;
    }
    // Local to global
    y.zero_entries()?;
    meles
        .mesh_dm
        .local_to_global(&meles.y_loc, petsc_rs::InsertMode::ADD_VALUES, &mut y)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Compute the diagonal of an operator via libCEED
// -----------------------------------------------------------------------------
pub(crate) fn get_diagonal_ceed<'a>(
    d: &mut petsc_rs::vector::Vector<'a>,
    meles: &&Meles,
) -> petsc_rs::Result<()> {
    // Get libCEED operator diagonal
    {
        let mut x_loc_view = meles.x_loc.view_mut()?;
        let x_loc_borrow = meles.x_loc_ceed.set_borrowed_slice(&mut x_loc_view)?;
        meles
            .ceed_op
            .linear_assemble_diagonal(&mut meles.x_loc_ceed)?;
    }
    // Local to global
    d.zero_entries()?;
    meles
        .mesh_dm
        .local_to_global(&meles.x_loc, petsc_rs::InsertMode::ADD_VALUES, &mut d)?;
    Ok(())
}
// -----------------------------------------------------------------------------
