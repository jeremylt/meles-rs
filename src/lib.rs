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
}

// -----------------------------------------------------------------------------
// Modules
// -----------------------------------------------------------------------------
pub(crate) mod dm;

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
#[derive(Debug)]
pub struct Meles {
    yml: String,
    ceed: libceed::Ceed,
    method: crate::MethodType,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl Drop for Meles {
    fn drop(&mut self) {
        // TODO: Eventually this will deallocate Ratel
    }
}

impl Meles {
    /// Returns a Meles context initialized with the specified yml filepath
    ///
    /// # arguments
    ///
    /// * `yml` - Filepath to specification yml
    ///
    /// ```
    /// let meles = meles::Meles::new("/path/to/yml.yml");
    /// ```
    pub fn new(yml: impl Into<String> + Clone) -> Self {
        // TODO: Verify yml path, initalized correct ceed
        Self {
            yml: yml.into().clone(),
            ceed: libceed::Ceed::init("/cpu/self"),
            method: crate::MethodType::BenchmarkProblem,
        }
    }

    /// Returns a PETSc DM initialized with the specified yml filepath
    ///
    /// # arguments
    ///
    /// * `method` - Filepath to specification yml
    ///
    /// ```
    /// # use meles::prelude::*;
    /// # fn main() -> meles::Result<()> {
    /// let mut meles = meles::Meles::new("/path/to/yml.yml");
    /// let dm = meles.create_dm(meles::MethodType::BenchmarkProblem)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_dm(mut self, method: crate::MethodType) -> Result<()> {
        self.method = method;
        match self.method {
            crate::MethodType::BenchmarkProblem => Ok(()), /* TODO: Build DM for BPs
                                                            * TODO: Ratel methods */
        }
    }
}

// -----------------------------------------------------------------------------
