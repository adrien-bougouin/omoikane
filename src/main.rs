#[macro_use]
extern crate rulinalg;
extern crate omoikane;

use omoikane::Model;
use omoikane::regression::LinearRegressionModel;

fn main() {
    let mut model = LinearRegressionModel::new();

    // f(x) = x
    model.fit_supervised_dataset(&vec!((vector!(1.0), 2.0),
                                       (vector!(2.0), 4.0),
                                       (vector!(3.0), 6.0),
                                       (vector!(4.0), 8.0),
                                       (vector!(5.0), 10.0),
                                       (vector!(6.0), 12.0)));
    // f(1.5) = 1.5
    println!("{}", model.predict(&vector!(1.5)));
}
