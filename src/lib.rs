pub mod datasets;
pub mod optimization;
pub mod regression;

mod traits;
pub use traits::Model;

#[macro_use]
extern crate rulinalg;
extern crate gnuplot;

#[cfg(test)]
#[macro_use]
extern crate approx;
