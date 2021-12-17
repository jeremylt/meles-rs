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
    let x_loc = meles.x_loc.borrow();
    let mut x_loc_ceed = meles.x_loc_ceed.borrow_mut();
    let mut y_loc = meles.y_loc.borrow_mut();
    let mut y_loc_ceed = meles.y_loc_ceed.borrow_mut();
    // Global to local
    meles
        .mesh_dm
        .borrow()
        .global_to_local(&x, petsc_rs::InsertMode::INSERT_VALUES, &mut x_loc)?;
    // Apply libCEED operator
    {
        let mut x_loc_view = x_loc.view()?;
        let x_loc_borrow = x_loc_ceed
            .set_borrowed_slice(&mut x_loc_view.as_slice().expect("failed to deref to slice"))
            .expect("failed to set borrowed slice");
        let mut y_loc_view = y_loc.view_mut()?;
        let y_loc_borrow = y_loc_ceed
            .set_borrowed_slice(&mut y_loc_view.as_slice().expect("failed to deref to slice"))
            .expect("failed to set borrowed slice");

        meles
            .ceed_op
            .borrow()
            .apply(&x_loc_ceed, &mut y_loc_ceed)
            .expect("failed to apply libCEED operator");
    }
    // Local to global
    y.zero_entries()?;
    meles
        .mesh_dm
        .borrow()
        .local_to_global(&y_loc, petsc_rs::InsertMode::ADD_VALUES, &mut y)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Compute the diagonal of an operator via libCEED
// -----------------------------------------------------------------------------
pub(crate) fn get_diagonal_ceed<'a>(
    d: &mut petsc_rs::vector::Vector<'a>,
    meles: &&Meles,
) -> petsc_rs::Result<()> {
    let mut x_loc = meles.x_loc.borrow_mut();
    let mut x_loc_ceed = meles.x_loc_ceed.borrow_mut();
    // Get libCEED operator diagonal
    {
        let mut x_loc_view = x_loc.view_mut()?;
        let x_loc_borrow = x_loc_ceed
            .set_borrowed_slice(&mut x_loc_view.as_slice().expect("failed to deref to slice"))
            .expect("failed to set borrowed slice");

        meles
            .ceed_op
            .borrow()
            .linear_assemble_diagonal(&mut x_loc_ceed)
            .expect("failed to compute diagonal of libCEED operator");
    }
    // Local to global
    d.zero_entries()?;
    meles
        .mesh_dm
        .borrow()
        .local_to_global(&x_loc, petsc_rs::InsertMode::ADD_VALUES, &mut d)?;
    Ok(())
}
// -----------------------------------------------------------------------------