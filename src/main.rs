#[macro_use]
extern crate rulinalg;
extern crate gnuplot;

extern crate omoikane;

use gnuplot::{Figure, AxesCommon, Caption, Color};

use omoikane::Model;
use omoikane::regression::LinearRegressionModel;
use omoikane::datasets::nist_strd::linear_regression::norris;

fn main() {
    let inputs: Vec<f64> = norris().into_iter().map(|(x, _)| x.data().as_slice()[0]).collect();
    let labels: Vec<f64> = norris().into_iter().map(|(_, y)| y).collect();
    let mut model = LinearRegressionModel::new(0.000001, 200000);
    let mut figure = Figure::new();

    model.fit_supervised_dataset(&inputs.iter()
                                        .zip(labels.iter())
                                        .map(|(x, y)| (vector!(*x), *y))
                                        .collect());

    figure.set_terminal("png", "result.png")
          .axes2d()
          .set_x_label("x", &[])
          .set_y_label("y", &[])
          .lines(&inputs, labels, &[Caption("sample"), Color("blue")])
          .lines(&inputs, inputs.iter().map(|x| model.predict(&vector!(*x))), &[Caption("prediction"), Color("red")]);
    figure.show();
}
