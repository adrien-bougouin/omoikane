#[macro_use]
extern crate rulinalg;
extern crate gnuplot;

extern crate omoikane;

use gnuplot::{Figure, AxesCommon, Caption, Color};

use omoikane::Model;
use omoikane::regression::LinearRegressionModel;

fn main() {
    let mut model = LinearRegressionModel::new();
    let mut fg = Figure::new();
    let inputs = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let labels = vec!(-2.0, -4.0, -6.0, -8.0, -10.0, -12.0);

    // f(x) = -2x
    model.fit_supervised_dataset(&inputs.iter()
                                        .zip(labels.iter())
                                        .map(|(x, y)| (vector!(*x), *y))
                                        .collect());

    fg.set_terminal("png", "result.png")
      .axes2d()
      .set_x_label("x", &[])
      .set_y_label("y", &[])
      .lines(&inputs, labels, &[Caption("sample"), Color("blue")])
      .lines(&inputs, inputs.iter().map(|x| model.predict(&vector!(*x))), &[Caption("prediction"), Color("red")]);
    fg.show();
}
