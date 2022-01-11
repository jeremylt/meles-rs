// Fenced `rust` code blocks included from README.md are executed as part of doctests.
#![doc = include_str!("../README.md")]

// -----------------------------------------------------------------------------
// Crate prelude
// -----------------------------------------------------------------------------
use crate::prelude::*;

pub mod prelude {
    pub use crate::{Meles, MethodType};
    pub(crate) use libceed::prelude::*;
    pub(crate) use mpi;
    pub(crate) use petsc_rs::prelude::*;
    pub(crate) use std::cell::RefCell;
    pub(crate) use std::fmt;
}

// -----------------------------------------------------------------------------
// Modules
// -----------------------------------------------------------------------------
pub(crate) mod ceed_bps;
pub(crate) mod dm;
pub(crate) mod petsc_ops;

// -----------------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------------
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

impl From<petsc_rs::Error> for Error {
    fn from(petsc_error: petsc_rs::Error) -> Self {
        Self {
            message: petsc_error.to_string(),
        }
    }
}

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// This enum is used to specify if a Benchmark problem or (eventually) Ratel
/// problem is being solved
pub enum MethodType {
    BenchmarkProblem,
}

// -----------------------------------------------------------------------------
// Meles context
// -----------------------------------------------------------------------------
pub struct Meles<'a> {
    pub(crate) petsc: &'a petsc_rs::Petsc,
    pub(crate) ceed: libceed::Ceed,
    pub(crate) is_initalized: bool,
    pub(crate) method: crate::MethodType,
    pub(crate) mesh_dm: RefCell<petsc_rs::dm::DM<'a, 'a>>,
    pub(crate) x_loc: RefCell<petsc_rs::vector::Vector<'a>>,
    pub(crate) y_loc: RefCell<petsc_rs::vector::Vector<'a>>,
    pub(crate) x_loc_ceed: RefCell<Vector<'a>>,
    pub(crate) y_loc_ceed: RefCell<Vector<'a>>,
    pub(crate) op_ceed: RefCell<CompositeOperator<'a>>,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for Meles<'a> {
    fn drop(&mut self) {
        // TODO: Eventually this will deallocate Ratel
    }
}

impl<'a> Meles<'a> {
    /// Returns a Meles context initialized with the specified yml filepath
    ///
    /// # arguments
    ///
    /// * `yml` - Filepath to specification yml
    ///
    /// ```
    /// # use meles::prelude::*;
    /// # use petsc_rs::prelude::*;
    /// # fn main() -> meles::Result<()> {
    /// let petsc = petsc_rs::Petsc::init_no_args()?;
    /// let meles = meles::Meles::new(&petsc, "./examples/meles.yml")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(petsc: &'a petsc_rs::Petsc, yml: impl Into<String> + Clone) -> Result<Self> {
        // Insert yaml into options database
        let yml = yml.into().clone();
        petsc.options_insert_file(&yml)?;

        // Get ceed resource
        struct Opt {
            ceed_resource: String,
        }
        impl PetscOpt for Opt {
            fn from_petsc_opt_builder(pob: &mut PetscOptBuilder) -> petsc_rs::Result<Self> {
                let ceed_resource =
                    pob.options_string("-ceed", "Ceed resource specifier", "", "/cpu/self")?;
                Ok(Opt { ceed_resource })
            }
        }
        let Opt { ceed_resource } = petsc.options_get()?;

        // Return self
        let ceed = libceed::Ceed::init(&ceed_resource);
        let x_loc_ceed = ceed.vector(1)?;
        let y_loc_ceed = ceed.vector(1)?;
        let op_ceed = ceed.composite_operator()?;
        Ok(Self {
            petsc: &petsc,
            ceed: ceed,
            is_initalized: false,
            method: crate::MethodType::BenchmarkProblem,
            mesh_dm: RefCell::new(petsc_rs::dm::DM::plex_create(petsc.world())?),
            x_loc: RefCell::new(petsc_rs::vector::Vector::create(petsc.world())?),
            y_loc: RefCell::new(petsc_rs::vector::Vector::create(petsc.world())?),
            x_loc_ceed: RefCell::new(x_loc_ceed),
            y_loc_ceed: RefCell::new(y_loc_ceed),
            op_ceed: RefCell::new(op_ceed),
        })
    }

    /// Returns a PETSc DM initialized with the specified yml filepath
    ///
    /// # arguments
    ///
    /// * `method` - Filepath to specification yml
    ///
    /// ```
    /// # use meles::prelude::*;
    /// # use petsc_rs::prelude::*;
    /// # fn main() -> meles::Result<()> {
    /// let petsc = petsc_rs::Petsc::init_no_args()?;
    /// let mut meles = meles::Meles::new(&petsc, "./examples/meles.yml")?;
    /// let dm = meles.dm(meles::MethodType::BenchmarkProblem)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn dm(&mut self, method: crate::MethodType) -> Result<()> {
        self.method = method;
        match self.method {
            crate::MethodType::BenchmarkProblem => crate::ceed_bps::create_dm(self)?,
            // TODO: Ratel methods
        }

        Ok(())
    }

    /// Return a PETSc MatShell for the DM that uses a libCEED operator
    ///
    /// # arguments
    ///
    /// * `dm` - DM for the MatShell
    ///
    /// ```
    /// # use meles::prelude::*;
    /// # use petsc_rs::prelude::*;
    /// # fn main() -> meles::Result<()> {
    /// let petsc = petsc_rs::Petsc::init_no_args()?;
    /// let mut meles = meles::Meles::new(&petsc, "./examples/meles.yml")?;
    /// let dm = meles.dm(meles::MethodType::BenchmarkProblem)?;
    /// let mat = meles.mat_shell_from_dm()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn mat_shell_from_dm(&self) -> Result<petsc_rs::mat::MatShell<'a, 'a, &crate::Meles>> {
        // Check setup
        assert!(self.is_initalized, "must create dm before setting up mat");
        assert!(
            self.method == crate::MethodType::BenchmarkProblem,
            "only supported for BenchmarkProblems"
        );

        // Create MatShell from DM
        let mut mat = self
            .mesh_dm
            .borrow()
            .create_matrix()?
            .into_shell(Box::new(self))?;

        // Set operations
        mat.shell_set_operation_mvv(MatOperation::MATOP_MULT, |m, x, y| {
            let ctx = m.get_mat_data().unwrap();
            crate::petsc_ops::apply_local_ceed_op(x, y, ctx)?;
            Ok(())
        })?;
        mat.shell_set_operation_mv(MatOperation::MATOP_GET_DIAGONAL, |m, d| {
            let ctx = m.get_mat_data().unwrap();
            crate::petsc_ops::get_diagonal_ceed(d, ctx)?;
            Ok(())
        })?;

        Ok(mat)
    }
}

// -----------------------------------------------------------------------------
