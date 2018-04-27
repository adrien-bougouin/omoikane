extern crate omoikane;

use omoikane::Model;
use omoikane::regression::linear_regression::LinearRegressionModel;

fn main() {
    let model = LinearRegressionModel {};

    model.fit();
    model.predict()
}
