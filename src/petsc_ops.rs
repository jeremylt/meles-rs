use crate::prelude::*;

// -----------------------------------------------------------------------------
// Apply the local action of a libCEED operator and store result in PETSc vector
//   i.e. compute A X = Y
// -----------------------------------------------------------------------------
pub(crate) fn apply_local_ceed_op<'a>(
    x: &petsc::vector::Vector<'a>,
    y: &mut petsc::vector::Vector<'a>,
    context: &crate::MelesMatShellContext,
) -> petsc::Result<()> {
    let mut x_loc = context.x_loc.borrow_mut();
    let mut x_loc_ceed = context.x_loc_ceed.borrow_mut();
    let mut y_loc = context.y_loc.borrow_mut();
    let mut y_loc_ceed = context.y_loc_ceed.borrow_mut();
    // Global to local
    context
        .dm
        .borrow()
        .global_to_local(x, InsertMode::INSERT_VALUES, &mut x_loc)?;
    // Apply libCEED operator
    {
        let mut x_loc_view = x_loc.view_mut()?;
        let mut x_loc_view_slice = x_loc_view.as_slice_mut().expect("failed to deref to slice");
        let _x_loc_wrapper = x_loc_ceed
            .wrap_slice_mut(&mut x_loc_view_slice)
            .expect("failed to wrap slice");
        let mut y_loc_view = y_loc.view_mut()?;
        let mut y_loc_view_slice = y_loc_view.as_slice_mut().expect("failed to deref to slice");
        let _y_loc_wrapper = y_loc_ceed
            .wrap_slice_mut(&mut y_loc_view_slice)
            .expect("failed to wrap slice");

        context
            .op_ceed
            .borrow()
            .apply(&x_loc_ceed, &mut y_loc_ceed)
            .expect("failed to apply libCEED operator");
    }
    // Local to global
    y.zero_entries()?;
    context
        .dm
        .borrow()
        .local_to_global(&y_loc, InsertMode::ADD_VALUES, y)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Compute the diagonal of an operator via libCEED
// -----------------------------------------------------------------------------
pub(crate) fn compute_diagonal_ceed<'a>(
    d: &mut petsc::vector::Vector<'a>,
    context: &crate::MelesMatShellContext,
) -> petsc::Result<()> {
    let mut x_loc = context.x_loc.borrow_mut();
    let mut x_loc_ceed = context.x_loc_ceed.borrow_mut();
    // Get libCEED operator diagonal
    {
        let mut x_loc_view = x_loc.view_mut()?;
        let mut x_loc_view_slice = x_loc_view.as_slice_mut().expect("failed to deref to slice");
        let _x_loc_wrapper = x_loc_ceed
            .wrap_slice_mut(&mut x_loc_view_slice)
            .expect("failed to wrap slice");

        context
            .op_ceed
            .borrow()
            .linear_assemble_diagonal(&mut x_loc_ceed)
            .expect("failed to compute diagonal of libCEED operator");
    }
    // Local to global
    d.zero_entries()?;
    context
        .dm
        .borrow()
        .local_to_global(&x_loc, InsertMode::ADD_VALUES, d)?;
    Ok(())
}
// -----------------------------------------------------------------------------
