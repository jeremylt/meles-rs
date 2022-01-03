use crate::prelude::*;

// -----------------------------------------------------------------------------
// BP enum
// -----------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeedBP {
    BP1 = 1,
    BP2 = 2,
    BP3 = 3,
    BP4 = 4,
    BP5 = 5,
    BP6 = 6,
}

impl std::str::FromStr for CeedBP {
    type Err = crate::Error;
    fn from_str(s: &str) -> crate::Result<CeedBP> {
        match s {
            "bp1" => Ok(CeedBP::BP1),
            "bp2" => Ok(CeedBP::BP2),
            "bp3" => Ok(CeedBP::BP3),
            "bp4" => Ok(CeedBP::BP4),
            "bp5" => Ok(CeedBP::BP5),
            "bp6" => Ok(CeedBP::BP6),
            _ => Err(crate::Error {
                message: "failed to parse problem option".to_string(),
            }),
        }
    }
}

impl std::fmt::Display for CeedBP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ceed Benchmark Problem {}", self)
    }
}

// -----------------------------------------------------------------------------
// BPData struct for each problem
// -----------------------------------------------------------------------------
pub(crate) struct BPData {
    num_components: usize,
    q_data_size: usize,
    setup_name: String,
    apply_name: String,
    input_name: String,
    output_name: String,
    q_mode: libceed::QuadMode,
    set_boundary_conditions: bool,
}

pub(crate) fn get_bp_data(bp: CeedBP) -> crate::Result<BPData> {
    match bp {
        CeedBP::BP1 => Ok(BPData {
            num_components: 1,
            q_data_size: 1,
            setup_name: "Mass3DBuild".to_string(),
            apply_name: "MassApply".to_string(),
            input_name: "u".to_string(),
            output_name: "v".to_string(),
            q_mode: libceed::QuadMode::Gauss,
            set_boundary_conditions: false,
        }),
        CeedBP::BP2 => Ok(BPData {
            num_components: 3,
            q_data_size: 1,
            setup_name: "Mass3DBuild".to_string(),
            apply_name: "Vector3MassApply".to_string(),
            input_name: "u".to_string(),
            output_name: "v".to_string(),
            q_mode: libceed::QuadMode::Gauss,
            set_boundary_conditions: false,
        }),
        CeedBP::BP3 => Ok(BPData {
            num_components: 1,
            q_data_size: 6,
            setup_name: "Poisson3DBuild".to_string(),
            apply_name: "Poisson3DApply".to_string(),
            input_name: "du".to_string(),
            output_name: "dv".to_string(),
            q_mode: libceed::QuadMode::Gauss,
            set_boundary_conditions: true,
        }),
        CeedBP::BP4 => Ok(BPData {
            num_components: 3,
            q_data_size: 6,
            setup_name: "Poisson3DBuild".to_string(),
            apply_name: "Vector3Poisson3DApply".to_string(),
            input_name: "du".to_string(),
            output_name: "dv".to_string(),
            q_mode: libceed::QuadMode::Gauss,
            set_boundary_conditions: true,
        }),
        CeedBP::BP5 => Ok(BPData {
            num_components: 1,
            q_data_size: 6,
            setup_name: "Poisson3DBuild".to_string(),
            apply_name: "Poisson3DApply".to_string(),
            input_name: "du".to_string(),
            output_name: "dv".to_string(),
            q_mode: libceed::QuadMode::GaussLobatto,
            set_boundary_conditions: true,
        }),
        CeedBP::BP6 => Ok(BPData {
            num_components: 3,
            q_data_size: 6,
            setup_name: "Poisson3DBuild".to_string(),
            apply_name: "Vector3Poisson3DApply".to_string(),
            input_name: "du".to_string(),
            output_name: "dv".to_string(),
            q_mode: libceed::QuadMode::GaussLobatto,
            set_boundary_conditions: true,
        }),
    }
}

// -----------------------------------------------------------------------------
// Setup dm and libCEED operator
// -----------------------------------------------------------------------------
pub(crate) fn create_dm(meles: crate::Meles) -> crate::Result<()> {
    // Get command line options
    struct Opt {
        mesh_file: String,
        problem: CeedBP,
        order: petsc_rs::PetscInt,
        q_extra: petsc_rs::PetscInt,
        faces: (petsc_rs::PetscInt, petsc_rs::PetscInt, petsc_rs::PetscInt),
    }
    impl PetscOpt for Opt {
        fn from_petsc_opt_builder(pob: &mut PetscOptBuilder) -> petsc_rs::Result<Self> {
            let mesh_file = pob.options_string("-mesh", "Read mesh from file", "", "")?;
            let problem = pob.options_from_string(
                "-problem",
                "CEED benchmark problem to solve",
                "",
                CeedBP::BP1,
            )?;
            let order =
                pob.options_int("-order", "Polynomial order of tensor product basis", "", 3)?;
            let q_extra = pob.options_int("-qextra", "Number of extra quadrature points", "", 1)?;
            let faces = (3, 3, 3);
            Ok(Opt {
                mesh_file,
                problem,
                order,
                q_extra,
                faces,
            })
        }
    }
    let Opt {
        mesh_file,
        problem,
        order,
        q_extra,
        faces,
    } = meles.petsc.options_get()?;
    let BPData {
        num_components,
        q_data_size,
        setup_name,
        apply_name,
        input_name,
        output_name,
        q_mode,
        set_boundary_conditions,
    } = get_bp_data(problem)?;

    // Create DM
    let dim: usize = 3;
    let p: usize = order as usize + 1;
    let q: usize = p + q_extra as usize;
    let is_simplex = false;
    let interpolate = true;
    let mut mesh_dm = if mesh_file != "" {
        petsc_rs::dm::DM::plex_create_from_file(meles.petsc.world(), mesh_file, interpolate)?
    } else {
        petsc_rs::dm::DM::plex_create_box_mesh(
            meles.petsc.world(),
            dim as i32,
            is_simplex,
            faces,
            None,
            None,
            None,
            interpolate,
        )?
    };

    // Set boundaries, order
    let boundary_function_diff =
        |dim: petsc_rs::PetscInt,
         t: petsc_rs::PetscReal,
         x: &[petsc_rs::PetscReal],
         num_components: petsc_rs::PetscInt,
         u: &mut [petsc_rs::PetscScalar]| {
            let c = [0., 1., 2.];
            let k = [1., 2., 3.];
            for i in 0..num_components as usize {
                u[i] = (std::f64::consts::PI * (c[0] + k[0] * x[0])).sin()
                    * (std::f64::consts::PI * (c[1] + k[1] * x[1])).sin()
                    * (std::f64::consts::PI * (c[2] + k[2] * x[2])).sin();
            }
            Ok(())
        };
    let user_boundary_function = if set_boundary_conditions {
        Some(boundary_function_diff)
    } else {
        None
    };
    crate::dm::setup_dm_by_order(
        meles.petsc.world(),
        mesh_dm,
        order as petsc_rs::PetscInt,
        num_components as petsc_rs::PetscInt,
        dim as petsc_rs::PetscInt,
        set_boundary_conditions,
        user_boundary_function,
    )?;

    // Create work vectors
    meles.x_loc = RefCell::new(Some(mesh_dm.create_local_vector()?));
    meles.y_loc = RefCell::new(Some(mesh_dm.create_local_vector()?));
    meles.x_loc_ceed = RefCell::new(Some(
        meles
            .ceed
            .vector(meles.x_loc.borrow().unwrap().get_local_size()? as usize)?,
    ));
    meles.y_loc_ceed = RefCell::new(Some(
        meles
            .ceed
            .vector(meles.x_loc.borrow().unwrap().get_local_size()? as usize)?,
    ));
    meles.mesh_dm = RefCell::new(Some(mesh_dm));

    // Create libCEED operator
    // -- Restrictions
    // -- Vector
    let mut qdata = meles.ceed.vector(1)?;
    let mut coord_loc = meles.mesh_dm.borrow().unwrap().get_coordinates_local()?;
    let mut coord_loc_ceed = meles.ceed.vector(coord_loc.get_local_size()? as usize)?;
    // -- Basis
    let basis_x = meles
        .ceed
        .basis_tensor_H1_Lagrange(dim, dim, 2, q, q_mode)?;
    let basis_u = meles
        .ceed
        .basis_tensor_H1_Lagrange(dim, num_components, p, q, q_mode)?;
    // -- QFunction
    let qf_setup = meles.ceed.q_function_interior_by_name(&setup_name)?;
    let qf_apply = meles.ceed.q_function_interior_by_name(&apply_name)?;
    // -- Apply setup operator
    {
        let mut coord_loc_view = coord_loc.view()?;
        let coord_loc_wrapper = coord_loc_ceed
            .wrap_slice_mut(&mut coord_loc_view.as_slice().expect("failed to deref to slice"))
            .expect("failed to wrap slice");
        meles
            .ceed
            .operator(&qf_setup, QFunctionOpt::None, QFunctionOpt::None)?
            .field("dx", &restr_x, &basis_x, VectorOpt::Active)?
            .field(
                "weights",
                ElemRestrictionOpt::None,
                &basis_x,
                VectorOpt::None,
            )?
            .field(
                "qdata",
                &restr_qdata,
                BasisOpt::Collocated,
                VectorOpt::Active,
            )?
            .check()?
            .apply(&coord_loc_ceed, &mut qdata)?;
    }
    // -- Operator
    meles.ceed_op = RefCell::new(Some(
        meles
            .ceed
            .operator(&qf_apply, QFunctionOpt::None, QFunctionOpt::None)?
            .field(&input_name, &restr_u, &basis_u, VectorOpt::Active)?
            .field("qdata", &restr_qdata, BasisOpt::Collocated, &qdata)?
            .field(&output_name, &restr_u, &basis_u, VectorOpt::Active)?
            .check()?,
    ));

    Ok(())
}
// -----------------------------------------------------------------------------
