// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

use libceed::{prelude::*, Ceed};
use mpi;
use petsc_rs::prelude::*;
use std::fmt;

// ----------------------------------------------------------------------------
// Error handling
// ----------------------------------------------------------------------------
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl From<libceed::Error> for Error {
    fn from(ceed_error: libceed::Error) -> Self {
        Self {
            message: ceed_error.to_string(),
        }
    }
}

impl From<petsc_rs::PetscError> for Error {
    fn from(petsc_error: petsc_rs::PetscError) -> Self {
        Self {
            message: petsc_error.to_string(),
        }
    }
}

// -----------------------------------------------------------------------------
// Apply 3D Kershaw mesh transformation
// The eps parameters are in (0, 1]
// Uniform mesh is recovered for eps=1
// -----------------------------------------------------------------------------
pub fn kershaw_transformation<'a>(
    mut dm: petsc_rs::dm::DM<'a, 'a>,
    eps: PetscScalar,
) -> crate::Result<()> {
    // Transition from a value of "a" for x=0, to a value of "b" for x=1.  Optionally
    // smooth -- see the commented versions at the end.
    fn step(a: PetscScalar, b: PetscScalar, x: PetscScalar) -> PetscScalar {
        if x <= 0. {
            a
        } else if x >= 1. {
            b
        } else {
            a + (b - a) * (x)
        }
    }

    // 1D transformation at the right boundary
    fn right(eps: PetscScalar, x: PetscScalar) -> PetscScalar {
        if x <= 0.5 {
            (2. - eps) * x
        } else {
            1. + eps * (x - 1.)
        }
    }

    // 1D transformation at the left boundary
    fn left(eps: PetscScalar, x: PetscScalar) -> PetscScalar {
        1. - right(eps, 1. - x)
    }

    let mut coords = dm.get_coordinates_local()?;
    let num_coords = coords.get_local_size()?;
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
pub fn setup_dm_by_order<'a, BcFn>(
    comm: &'a mpi::topology::UserCommunicator,
    mut dm: petsc_rs::dm::DM<'a, 'a>,
    order: petsc_rs::PetscInt,
    num_components: petsc_rs::PetscInt,
    dimemsion: petsc_rs::PetscInt,
    enforce_boundary_conditions: bool,
    user_boundary_function: Option<BcFn>,
) -> crate::Result<()>
where
    BcFn: Fn(
            petsc_rs::PetscInt,
            petsc_rs::PetscReal,
            &[petsc_rs::PetscReal],
            petsc_rs::PetscInt,
            &mut [petsc_rs::PetscScalar],
        ) -> petsc_rs::Result<()>
        + 'a,
{
    // Setup FE
    let fe = petsc_rs::dm::FEDisc::create_lagrange(
        &comm,
        dimemsion,
        num_components,
        false,
        order,
        None,
    )?;
    dm.add_field(None, fe)?;

    // Setup DM
    let ds = dm.create_ds()?;
    if enforce_boundary_conditions {
        let has_label = dm.has_label("marker")?;
        if !has_label {
            dm.create_label("marker")?;
            let mut label = dm.get_label("marker")?.unwrap();
            dm.plex_mark_boundary_faces(1, &mut label)?;
        }
        let mut label = dm.get_label("marker")?.unwrap();
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

// ----------------------------------------------------------------------------
// Involute index - essential BC DoFs are encoded in closure incides as -(i+1)
// ----------------------------------------------------------------------------
pub(crate) fn involute(i: PetscInt) -> PetscInt {
    if i >= 0 {
        i
    } else {
        -(i + 1)
    }
}

// -----------------------------------------------------------------------------
// Setup Restriction from DMPlex
// -----------------------------------------------------------------------------
pub fn create_restriction_from_dm_plex(
    dm: petsc_rs::dm::DM,
    ceed: libceed::Ceed,
    p: petsc_rs::PetscInt,
    topological_dimension: petsc_rs::PetscInt,
    height: petsc_rs::PetscInt,
    label: petsc_rs::dm::DMLabel,
    value: petsc_rs::PetscInt,
) -> crate::Result<()> {
    Ok(())
}

// ----------------------------------------------------------------------------
