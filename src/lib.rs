#[macro_use]
extern crate rulinalg;
extern crate gnuplot;

pub mod optimization;
pub mod regression;

mod traits;
pub use traits::Model;

#[cfg(test)]
#[macro_use]
extern crate approx;
