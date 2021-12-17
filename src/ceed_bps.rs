use crate::prelude::*;

// TODO: Objects representing BP fields

// -----------------------------------------------------------------------------
// Setup Restriction from DMPlex
// -----------------------------------------------------------------------------
pub(crate) fn create_dm(meles: crate::Meles) -> crate::Result<()> {
    // Get command line options
    struct Opt {
        mesh_file: String,
        order: petsc_rs::PetscInt,
        q_extra: petsc_rs::PetscInt,
        faces: (petsc_rs::PetscInt, petsc_rs::PetscInt, petsc_rs::PetscInt),
    }
    impl PetscOpt for Opt {
        fn from_petsc_opt_builder(pob: &mut PetscOptBuilder) -> petsc_rs::Result<Self> {
            let mesh_file = pob.options_string("-mesh", "Read mesh from file", "", "")?;
            let order =
                pob.options_int("-order", "Polynomial order of tensor product basis", "", 3)?;
            let q_extra = pob.options_int("-qextra", "Number of extra quadrature points", "", 1)?;
            let faces = (3, 3, 3);
            Ok(Opt {
                mesh_file,
                order,
                q_extra,
                faces,
            })
        }
    }
    let Opt {
        mesh_file,
        order,
        q_extra,
        faces,
    } = meles.petsc.options_get()?;

    // Create DM
    let dim = 3;
    let is_simplex = false;
    let interpolate = true;
    let mut mesh_dm = if mesh_file != "" {
        petsc_rs::dm::DM::plex_create_from_file(meles.petsc.world(), mesh_file, interpolate)?
    } else {
        petsc_rs::dm::DM::plex_create_box_mesh(
            meles.petsc.world(),
            dim,
            is_simplex,
            faces,
            None,
            None,
            None,
            interpolate,
        )?
    };

    // Set boundaries, order
    let fe = petsc_rs::dm::FEDisc::create_lagrange(
        meles.petsc.world(),
        dim,
        num_comp,
        is_simplex,
        order,
        order + q_extra,
    )?;
    mesh_dm.add_field(None, fe)?;
    mesh_dm.plex_set_closure_permutation_tensor_default(None)?;

    // Create work vectors
    meles.x_loc = RefCell::new(mesh_dm.create_local_vector()?);
    meles.y_loc = RefCell::new(mesh_dm.create_local_vector()?);
    meles.x_loc_ceed = RefCell::new(
        meles
            .ceed
            .vector(meles.x_loc.borrow().get_local_size()? as usize)?,
    );
    meles.y_loc_ceed = RefCell::new(
        meles
            .ceed
            .vector(meles.x_loc.borrow().get_local_size()? as usize)?,
    );

    // Create libCEED operators

    // Set MatShell

    Ok(())
}
// -----------------------------------------------------------------------------
