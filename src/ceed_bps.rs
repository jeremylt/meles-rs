use crate::prelude::*;

// -----------------------------------------------------------------------------
// BP command line options
// -----------------------------------------------------------------------------
struct Opt {
    problem: CeedBP,
    order: usize,
    q_extra: usize,
}

impl petsc::Opt for Opt {
    fn from_opt_builder(pob: &mut petsc::OptBuilder) -> petsc::Result<Self> {
        let problem = pob.options_from_string(
            "-problem",
            "CEED benchmark problem to solve",
            "",
            CeedBP::BP1,
        )?;
        let order =
            pob.options_usize("-order", "Polynomial order of tensor product basis", "", 3)?;
        let q_extra = pob.options_usize("-qextra", "Number of extra quadrature points", "", 1)?;
        Ok(Opt {
            problem,
            order,
            q_extra,
        })
    }
}

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
        write!(f, "bp{}", *self as usize)
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

pub(crate) fn bp_data(bp: CeedBP) -> crate::Result<BPData> {
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

// Boundary function
pub(crate) fn boundary_function_diff(
    _dim: petsc::Int,
    _t: Real,
    x: &[Real],
    num_components: petsc::Int,
    u: &mut [petsc::Scalar],
) -> petsc::Result<()> {
    let c = [0., 1., 2.];
    let k = [1., 2., 3.];
    for i in 0..num_components as usize {
        u[i] = (std::f64::consts::PI * (c[0] + k[0] * x[0])).sin()
            * (std::f64::consts::PI * (c[1] + k[1] * x[1])).sin()
            * (std::f64::consts::PI * (c[2] + k[2] * x[2])).sin();
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Setup dm and libCEED operator
// -----------------------------------------------------------------------------
pub(crate) fn create_dm(petsc: &Petsc) -> crate::Result<DM<'_, '_>> {
    let Opt {
        problem,
        order,
        q_extra: _,
    } = petsc.options()?;
    let BPData {
        num_components,
        q_data_size: _,
        setup_name: _,
        apply_name: _,
        input_name: _,
        output_name: _,
        q_mode: _,
        set_boundary_conditions,
    } = bp_data(problem)?;

    // Create DM
    let mut dm = DM::create(petsc.world())?;
    dm.set_type(DMType::DMPLEX)?;
    dm.set_from_options()?;

    let user_boundary_function = if set_boundary_conditions {
        Some(boundary_function_diff)
    } else {
        None
    };
    crate::dm::setup_dm_by_order(
        &mut dm,
        order,
        num_components,
        set_boundary_conditions,
        user_boundary_function,
    )?;

    Ok(dm)
}

// -----------------------------------------------------------------------------
// Setup dm and libCEED operator
// -----------------------------------------------------------------------------
pub(crate) fn mat_shell_context<'a>(
    meles: &'a crate::Meles<'a>,
    petsc: &'a Petsc,
) -> crate::Result<crate::MelesMatShellContext<'a>> {
    let Opt {
        problem,
        order,
        q_extra,
    } = petsc.options()?;
    let BPData {
        num_components,
        q_data_size,
        setup_name,
        apply_name,
        input_name,
        output_name,
        q_mode,
        set_boundary_conditions,
    } = bp_data(problem)?;

    // Duplicate DM
    let mut dm = meles.dm.borrow().clone();
    let user_boundary_function = if set_boundary_conditions {
        Some(boundary_function_diff)
    } else {
        None
    };
    crate::dm::setup_dm_by_order(
        &mut dm,
        order,
        num_components,
        set_boundary_conditions,
        user_boundary_function,
    )?;

    // Create work vectors
    let x_loc = dm.create_local_vector()?;
    let y_loc = dm.create_local_vector()?;
    let x_loc_size = x_loc.local_size()?;
    let x_loc_ceed = meles.ceed.vector(x_loc_size)?;
    let y_loc_ceed = meles.ceed.vector(x_loc_size)?;

    // Create libCEED operator
    // -- Basis
    let p = order + 1;
    let q = p + q_extra;
    let dimension = dm.dimension()?;
    let basis_x = meles
        .ceed
        .basis_tensor_H1_Lagrange(dimension, dimension, 2, q, q_mode)?;
    let basis_u = meles
        .ceed
        .basis_tensor_H1_Lagrange(dimension, num_components, p, q, q_mode)?;
    // -- Restrictions
    let restr_u = crate::dm::create_restriction_from_dm_plex(&dm, &meles.ceed, 0, None, 0)?;
    let restr_x = {
        let mesh_coord_dm = dm.coordinate_dm()?;
        crate::dm::create_restriction_from_dm_plex(&mesh_coord_dm, &meles.ceed, 0, None, 0)?
    };
    let restr_qdata = {
        let num_elements = restr_u.num_elements();
        let num_quadrature_points = basis_u.num_quadrature_points();
        meles.ceed.strided_elem_restriction(
            num_elements,
            num_quadrature_points,
            q_data_size,
            num_elements * num_quadrature_points * q_data_size,
            CEED_STRIDES_BACKEND,
        )?
    };
    // -- Vector
    let mut qdata = restr_qdata.create_lvector()?;
    let mut coord_loc = {
        let mut dm = meles.dm.borrow_mut();
        dm.coordinates_local()?
    };
    let mut coord_loc_ceed = meles.ceed.vector(coord_loc.local_size()?)?;
    // -- QFunction
    let qf_setup = meles.ceed.q_function_interior_by_name(&setup_name)?;
    let qf_apply = meles.ceed.q_function_interior_by_name(&apply_name)?;
    // -- Apply setup operator
    {
        let mut coord_loc_view = coord_loc.view_mut()?;
        let mut coord_loc_view_slice = coord_loc_view
            .as_slice_mut()
            .expect("failed to deref to slice");
        let _coord_loc_wrapper = coord_loc_ceed
            .wrap_slice_mut(&mut coord_loc_view_slice)
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
    let op_ceed = meles
        .ceed
        .operator(&qf_apply, QFunctionOpt::None, QFunctionOpt::None)?
        .field(&input_name, &restr_u, &basis_u, VectorOpt::Active)?
        .field("qdata", &restr_qdata, BasisOpt::Collocated, &qdata)?
        .field(&output_name, &restr_u, &basis_u, VectorOpt::Active)?
        .check()?;

    // Return object
    Ok(crate::MelesMatShellContext {
        dm: RefCell::new(dm),
        x_loc: RefCell::new(x_loc),
        y_loc: RefCell::new(y_loc),
        x_loc_ceed: RefCell::new(x_loc_ceed),
        y_loc_ceed: RefCell::new(y_loc_ceed),
        op_ceed: RefCell::new(op_ceed),
    })
}

// -----------------------------------------------------------------------------
