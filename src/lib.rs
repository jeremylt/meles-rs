// Fenced `rust` code blocks included from README.md are executed as part of doctests.
#![doc = include_str!("../README.md")]

// -----------------------------------------------------------------------------
// Crate prelude
// -----------------------------------------------------------------------------
use crate::prelude::*;

pub mod prelude {
    pub use crate::{Meles, MelesMatShellContext, MethodType};
    pub(crate) use libceed::prelude::*;
    pub(crate) use petsc::prelude::*;
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

impl From<petsc::Error> for Error {
    fn from(petsc_error: petsc::Error) -> Self {
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
// Meles MatShell context
// -----------------------------------------------------------------------------
pub struct MelesMatShellContext<'a> {
    pub(crate) dm: RefCell<DM<'a, 'a>>,
    pub(crate) x_loc: RefCell<petsc::vector::Vector<'a>>,
    pub(crate) y_loc: RefCell<petsc::vector::Vector<'a>>,
    pub(crate) x_loc_ceed: RefCell<libceed::vector::Vector<'a>>,
    pub(crate) y_loc_ceed: RefCell<libceed::vector::Vector<'a>>,
    pub(crate) op_ceed: RefCell<libceed::operator::Operator<'a>>,
}

// -----------------------------------------------------------------------------
// Meles context
// -----------------------------------------------------------------------------
pub struct Meles<'a> {
    pub(crate) ceed: libceed::Ceed,
    pub(crate) method: crate::MethodType,
    pub dm: RefCell<DM<'a, 'a>>,
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
    /// * `petsc` - PETSc context to use
    /// * `yml` - Filepath to specification yml
    /// * `method` - Type of meles problem to setup
    ///
    /// ```
    /// # use meles::prelude::*;
    /// # use petsc::prelude::*;
    /// # fn main() -> meles::Result<()> {
    /// let petsc = petsc::Petsc::init_no_args()?;
    /// let mut meles = meles::Meles::new(
    ///     &petsc,
    ///     "./examples/meles.yml",
    ///     meles::MethodType::BenchmarkProblem,
    /// )?;
    ///
    /// // mesh DM can be borrowed immutably
    /// let vec = meles.dm.borrow().create_global_vector()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        petsc: &'a Petsc,
        yml: impl Into<String> + Clone,
        method: crate::MethodType,
    ) -> Result<Self> {
        // Insert yaml into options database
        let yml = yml.into().clone();
        petsc.options_insert_file(&yml)?;

        // Create Ceed
        struct Opt {
            ceed_resource: String,
        }
        impl petsc::Opt for Opt {
            fn from_opt_builder(pob: &mut petsc::OptBuilder) -> petsc::Result<Self> {
                let ceed_resource = pob.options_string(
                    "-ceed",
                    "libceed::Ceed resource specifier",
                    "",
                    "/cpu/self",
                )?;
                Ok(Opt { ceed_resource })
            }
        }
        let Opt { ceed_resource } = petsc.options()?;
        let ceed = libceed::Ceed::init(&ceed_resource);

        // Create DM
        let dm = match method {
            crate::MethodType::BenchmarkProblem => crate::ceed_bps::create_dm(&petsc)?,
            // TODO: Ratel methods
        };

        // Return self
        Ok(Self {
            ceed: ceed,
            method: crate::MethodType::BenchmarkProblem,
            dm: RefCell::new(dm),
        })
    }

    /// Return a PETSc MatShell for the DM that uses a libCEED operator
    ///
    /// Note: Can only directly create a MatShell for `BenchmarkProblem`s
    ///
    /// ```
    /// # use meles::prelude::*;
    /// # use petsc::prelude::*;
    /// # fn main() -> meles::Result<()> {
    /// let petsc = petsc::Petsc::init_no_args()?;
    /// let meles = meles::Meles::new(
    ///     &petsc,
    ///     "./examples/meles.yml",
    ///     meles::MethodType::BenchmarkProblem,
    /// )?;
    ///
    /// // create matshell
    /// let mat = meles.mat_shell(&petsc)?;
    /// let mut ksp = petsc.ksp_create()?;
    /// ksp.set_operators(&mat, &mat)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn mat_shell(
        &'a self,
        petsc: &'a Petsc,
    ) -> Result<petsc::mat::MatShell<'a, 'a, crate::MelesMatShellContext<'a>>> {
        // Check setup
        assert!(
            self.method == crate::MethodType::BenchmarkProblem,
            "only supported for BenchmarkProblems"
        );

        // Create MatShellContext
        let context = crate::ceed_bps::mat_shell_context(&self, &petsc)?;

        // Create MatShell from DM
        let mut mat = self
            .dm
            .borrow()
            .create_matrix()?
            .into_shell(Box::new(context))?;

        // Set operations
        mat.shell_set_operation_mvv(MatOperation::MATOP_MULT, |m, x, y| {
            let context = m.mat_data().unwrap();
            crate::petsc_ops::apply_local_ceed_op(x, y, context)?;
            Ok(())
        })?;
        mat.shell_set_operation_mv(MatOperation::MATOP_GET_DIAGONAL, |m, d| {
            let context = m.mat_data().unwrap();
            crate::petsc_ops::compute_diagonal_ceed(d, context)?;
            Ok(())
        })?;

        Ok(mat)
    }
}

// -----------------------------------------------------------------------------
