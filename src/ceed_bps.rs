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
    num_comp: usize,
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
            num_comp: 1,
            q_data_size: 1,
            setup_name: "Mass3DBuild".to_string(),
            apply_name: "MassApply".to_string(),
            input_name: "u".to_string(),
            output_name: "v".to_string(),
            q_mode: libceed::QuadMode::Gauss,
            set_boundary_conditions: false,
        }),
        CeedBP::BP2 => Ok(BPData {
            num_comp: 3,
            q_data_size: 1,
            setup_name: "Mass3DBuild".to_string(),
            apply_name: "Vector3MassApply".to_string(),
            input_name: "u".to_string(),
            output_name: "v".to_string(),
            q_mode: libceed::QuadMode::Gauss,
            set_boundary_conditions: false,
        }),
        CeedBP::BP3 => Ok(BPData {
            num_comp: 1,
            q_data_size: 6,
            setup_name: "Poisson3DBuild".to_string(),
            apply_name: "Poisson3DApply".to_string(),
            input_name: "du".to_string(),
            output_name: "dv".to_string(),
            q_mode: libceed::QuadMode::Gauss,
            set_boundary_conditions: true,
        }),
        CeedBP::BP4 => Ok(BPData {
            num_comp: 3,
            q_data_size: 6,
            setup_name: "Poisson3DBuild".to_string(),
            apply_name: "Vector3Poisson3DApply".to_string(),
            input_name: "du".to_string(),
            output_name: "dv".to_string(),
            q_mode: libceed::QuadMode::Gauss,
            set_boundary_conditions: true,
        }),
        CeedBP::BP5 => Ok(BPData {
            num_comp: 1,
            q_data_size: 6,
            setup_name: "Poisson3DBuild".to_string(),
            apply_name: "Poisson3DApply".to_string(),
            input_name: "du".to_string(),
            output_name: "dv".to_string(),
            q_mode: libceed::QuadMode::GaussLobatto,
            set_boundary_conditions: true,
        }),
        CeedBP::BP6 => Ok(BPData {
            num_comp: 3,
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
        num_comp,
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
    let fe = petsc_rs::dm::FEDisc::create_lagrange(
        meles.petsc.world(),
        dim as i32,
        num_comp as i32,
        is_simplex,
        p as i32,
        q as i32,
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
    meles.mesh_dm = RefCell::new(mesh_dm);

    // Create libCEED operator
    // -- Restrictions
    // -- Vector
    let mut qdata = meles.ceed.vector(1)?;
    // -- Basis
    let basis_x = meles
        .ceed
        .basis_tensor_H1_Lagrange(dim, dim, 2, q, q_mode)?;
    let basis_u = meles
        .ceed
        .basis_tensor_H1_Lagrange(dim, num_comp, p, q, q_mode)?;
    // -- Restriction

    // -- QFunction
    let qf_setup = meles.ceed.q_function_interior_by_name(&setup_name)?;
    let qf_apply = meles.ceed.q_function_interior_by_name(&apply_name)?;
    // -- Apply setup operator
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
        .apply(&x, &mut qdata)?;
    // -- Operator
    meles.ceed_op = RefCell::new(
        meles
            .ceed
            .operator(&qf_apply, QFunctionOpt::None, QFunctionOpt::None)?
            .field(&input_name, &restr_u, &basis_u, VectorOpt::Active)?
            .field("qdata", &restr_qdata, BasisOpt::Collocated, &qdata)?
            .field(&output_name, &restr_u, &basis_u, VectorOpt::Active)?
            .check()?,
    );

    Ok(())
}
// -----------------------------------------------------------------------------
