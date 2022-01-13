use crate::prelude::*;

// -----------------------------------------------------------------------------
// Apply 3D Kershaw mesh transformation
// The eps parameters are in (0, 1]
// Uniform mesh is recovered for eps=1
// -----------------------------------------------------------------------------
pub(crate) fn kershaw_transformation<'a>(
    mut dm: DM<'a, 'a>,
    eps: petsc::Scalar,
) -> crate::Result<()> {
    // Transition from a value of "a" for x=0, to a value of "b" for x=1.  Optionally
    // smooth -- see the commented versions at the end.
    fn step(a: petsc::Scalar, b: petsc::Scalar, x: petsc::Scalar) -> petsc::Scalar {
        if x <= 0. {
            a
        } else if x >= 1. {
            b
        } else {
            a + (b - a) * (x)
        }
    }

    // 1D transformation at the right boundary
    fn right(eps: petsc::Scalar, x: petsc::Scalar) -> petsc::Scalar {
        if x <= 0.5 {
            (2. - eps) * x
        } else {
            1. + eps * (x - 1.)
        }
    }

    // 1D transformation at the left boundary
    fn left(eps: petsc::Scalar, x: petsc::Scalar) -> petsc::Scalar {
        1. - right(eps, 1. - x)
    }

    let mut coords = dm.coordinates_local()?;
    let num_coords = coords.local_size()?;
    let mut coord_view = coords.view_mut()?;

    // Apply transformations based upon layer
    for i in (0..num_coords as usize).step_by(3) {
        let (x, y, z) = (coord_view[i], coord_view[i + 1], coord_view[i + 2]);
        let layer = 6 * x as i32;
        let lambda = (x - layer as f64 / 6.0) * 6.0;

        match layer {
            0 => {
                coord_view[i + 1] = left(eps, y);
                coord_view[i + 2] = left(eps, z);
            }
            1 | 4 => {
                coord_view[i + 1] = step(left(eps, y), right(eps, y), lambda);
                coord_view[i + 2] = step(left(eps, z), right(eps, z), lambda);
            }
            2 => {
                coord_view[i + 1] = step(right(eps, y), left(eps, y), lambda / 2.0);
                coord_view[i + 2] = step(right(eps, z), left(eps, z), lambda / 2.0);
            }
            3 => {
                coord_view[i + 1] = step(right(eps, y), left(eps, y), (1.0 + lambda) / 2.0);
                coord_view[i + 2] = step(right(eps, z), left(eps, z), (1.0 + lambda) / 2.0);
            }
            _ => {
                coord_view[i + 1] = right(eps, y);
                coord_view[i + 2] = right(eps, z);
            }
        }
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
pub(crate) fn setup_dm_by_order<'a, BcFn>(
    comm: &'a mpi::topology::UserCommunicator,
    dm: &mut DM<'a, 'a>,
    order: usize,
    num_components: usize,
    dimemsion: usize,
    enforce_boundary_conditions: bool,
    user_boundary_function: Option<BcFn>,
) -> crate::Result<()>
where
    BcFn: Fn(petsc::Int, Real, &[Real], petsc::Int, &mut [petsc::Scalar]) -> petsc::Result<()> + 'a,
{
    // Setup FE
    let fe = FEDisc::create_lagrange(&comm, dimemsion, num_components, false, order, None)?;
    dm.add_field(None, fe)?;

    // Coordinate FE
    let fe_coords = FEDisc::create_lagrange(&comm, dimemsion, dimemsion, false, 1, None)?;
    dm.project_coordinates(fe_coords)?;

    // Setup DM
    let _ = dm.create_ds()?;
    if enforce_boundary_conditions {
        let has_label = dm.has_label("marker")?;
        if !has_label {
            dm.create_label("marker")?;
            let mut label = dm.label("marker")?.unwrap();
            dm.plex_mark_boundary_faces(1, &mut label)?;
        }
        let mut label = dm.label("marker")?.unwrap();
        dm.add_boundary_essential(
            "wall",
            &mut label,
            &[],
            1,
            &[],
            user_boundary_function.unwrap(),
        )?;
    }
    dm.plex_set_closure_permutation_tensor_default(None)?;

    Ok(())
}

// -----------------------------------------------------------------------------
// Setup Restriction from DMPlex
// -----------------------------------------------------------------------------
pub(crate) fn create_restriction_from_dm_plex<'a, 'b, 'c>(
    dm: &'a DM<'b, '_>,
    ceed: &libceed::Ceed,
    height: usize,
    label: impl Into<Option<&'b DMLabel<'b>>>,
    value: usize,
) -> crate::Result<ElemRestriction<'c>> {
    let DMPlexLocalOffsets {
        num_cells,
        cell_size,
        num_components,
        l_size,
        offsets,
    } = dm.plex_local_offsets(label, value, height, 0)?;
    let elem_restriction = ceed.elem_restriction(
        num_cells,
        cell_size,
        num_components,
        1,
        l_size,
        MemType::Host,
        &offsets,
    )?;
    Ok(elem_restriction)
}

// -----------------------------------------------------------------------------
