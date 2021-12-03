// Fenced `rust` code blocks included from README.md are executed as part of doctests.
#![doc = include_str!("../README.md")]

// -----------------------------------------------------------------------------
// Crate prelude
// -----------------------------------------------------------------------------
use crate::prelude::*;

pub mod prelude {
    pub(crate) use crate::dm;
    pub use crate::{Meles, MethodType};
    pub(crate) use libceed::{prelude::*, Ceed};
    pub(crate) use mpi;
    pub(crate) use petsc_rs::prelude::*;
    pub(crate) use std::fmt;
    pub(crate) use std::mem;
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

impl From<petsc_rs::PetscError> for Error {
    fn from(petsc_error: petsc_rs::PetscError) -> Self {
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
    petsc: &'a petsc_rs::Petsc,
    ceed: libceed::Ceed,
    method: crate::MethodType,
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
                    pob.options_string("-ceed", "ceed resource", "", "/cpu/self")?;
                Ok(Opt { ceed_resource })
            }
        }
        let Opt { ceed_resource } = petsc.options_get()?;

        // Return self
        Ok(Self {
            petsc: &petsc,
            ceed: libceed::Ceed::init(&ceed_resource),
            method: crate::MethodType::BenchmarkProblem,
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
    /// let meles = meles::Meles::new(&petsc, "./examples/meles.yml")?;
    /// let dm = meles.dm(meles::MethodType::BenchmarkProblem)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn dm(mut self, method: crate::MethodType) -> Result<()> {
        self.method = method;
        match self.method {
            crate::MethodType::BenchmarkProblem => crate::ceed_bps::create_dm(self), /* TODO: Build DM for BPs
                                                                                      * TODO: Ratel methods */
        }
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
    /// let meles = meles::Meles::new(&petsc, "./examples/meles.yml")?;
    /// let dm = meles.dm(meles::MethodType::BenchmarkProblem)?;
    /// let mat = meles.mat_shell_from_dm(dm)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn mat_shell_from_dm(
        &self,
        dm: petsc_rs::dm::DM<'a, 'a>,
    ) -> Result<petsc_rs::mat::MatShell<'a, 'a, &'a crate::Meles>> {
        // Create MatShell from DM
        let mut mat = dm.create_matrix()?.into_shell(Box::new(self))?;

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
