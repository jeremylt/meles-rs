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
    let x_loc = meles.x_loc.borrow().unwrap();
    let mut x_loc_ceed = meles.x_loc_ceed.borrow_mut().unwrap();
    let mut y_loc = meles.y_loc.borrow_mut().unwrap();
    let mut y_loc_ceed = meles.y_loc_ceed.borrow_mut().unwrap();
    // Global to local
    meles.mesh_dm.borrow().unwrap().global_to_local(
        x,
        petsc_rs::InsertMode::INSERT_VALUES,
        &mut x_loc,
    )?;
    // Apply libCEED operator
    {
        let mut x_loc_view = x_loc.view()?;
        let _x_loc_wrapper = x_loc_ceed
            .wrap_slice_mut(&mut x_loc_view.as_slice().expect("failed to deref to slice"))
            .expect("failed to wrap slice");
        let mut y_loc_view = y_loc.view_mut()?;
        let _y_loc_wrapper = y_loc_ceed
            .wrap_slice_mut(&mut y_loc_view.as_slice().expect("failed to deref to slice"))
            .expect("failed to wrap slice");

        meles
            .ceed_op
            .borrow()
            .as_ref()
            .unwrap()
            .apply(&x_loc_ceed, &mut y_loc_ceed)
            .expect("failed to apply libCEED operator");
    }
    // Local to global
    y.zero_entries()?;
    meles
        .mesh_dm
        .borrow()
        .unwrap()
        .local_to_global(&y_loc, petsc_rs::InsertMode::ADD_VALUES, y)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Compute the diagonal of an operator via libCEED
// -----------------------------------------------------------------------------
pub(crate) fn get_diagonal_ceed<'a>(
    d: &mut petsc_rs::vector::Vector<'a>,
    meles: &&Meles,
) -> petsc_rs::Result<()> {
    let mut x_loc = meles.x_loc.borrow_mut().unwrap();
    let mut x_loc_ceed = meles.x_loc_ceed.borrow_mut().unwrap();
    // Get libCEED operator diagonal
    {
        let mut x_loc_view = x_loc.view_mut()?;
        let _x_loc_wrapper = x_loc_ceed
            .wrap_slice_mut(&mut x_loc_view.as_slice().expect("failed to deref to slice"))
            .expect("failed to wrap slice");

        meles
            .ceed_op
            .borrow()
            .as_ref()
            .unwrap()
            .linear_assemble_diagonal(&mut x_loc_ceed)
            .expect("failed to compute diagonal of libCEED operator");
    }
    // Local to global
    d.zero_entries()?;
    meles
        .mesh_dm
        .borrow()
        .unwrap()
        .local_to_global(&x_loc, petsc_rs::InsertMode::ADD_VALUES, d)?;
    Ok(())
}
// -----------------------------------------------------------------------------
