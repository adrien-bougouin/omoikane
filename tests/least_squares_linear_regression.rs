extern crate omoikane;
extern crate rulinalg;
#[macro_use]
extern crate approx;

use omoikane::optimization::ParametricFunction;
use omoikane::optimization::LinearFunction;
use omoikane::optimization::least_squares_fit;
use omoikane::datasets::nist_strd::linear_regression::norris;

#[test]
fn least_squares_fit_linear_function_on_norris_dataset() {
    let mut function = LinearFunction::new(1);

    least_squares_fit(&mut function, &norris(), 0.000001, 200000);

    let ref parameters = function.parameters();

    assert_eq!(2, parameters.size());
    assert_relative_eq!(function.parameters().data().as_slice()[0],
                        -0.262323073774029,
                        epsilon = 0.232818234301152);
    assert_relative_eq!(function.parameters().data().as_slice()[1],
                        1.00211681802045,
                        epsilon = 0.00429796848199937);
}
