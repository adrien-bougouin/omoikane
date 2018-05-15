use rulinalg::vector::Vector;
use gnuplot::{Figure, AxesCommon, Color};

use super::ParametricFunction;

pub fn gradient_descent_fit<F>(dataset: &Vec<(Vector<f64>, f64)>,
                               parametric_function: &mut F,
                               compute_error_average: &Fn(&F, &Vec<(Vector<f64>, f64)>) -> f64,
                               compute_error_gradients: &Fn(&F, &Vec<(Vector<f64>, f64)>) -> Vector<f64>,
                               learning_rate: f64,
                               max_iterations: u32) -> Vec<f64>
where F: ParametricFunction {
    let mut errors = vec!();

    for _ in 1..max_iterations {
        let new_parameters = {
            let parameters = parametric_function.parameters();
            let gradients = compute_error_gradients(parametric_function, dataset);

            parameters - (gradients * learning_rate)
        };

        errors.push(compute_error_average(parametric_function, dataset));
        parametric_function.set_parameters(new_parameters);
    }

    // TODO: remove or debug mode
    {
        let x: Vec<u32> = (1..max_iterations).collect();
        let ref y = errors;
        let mut fg = Figure::new();
        fg.set_terminal("png", "error.png")
          .axes2d()
          .set_x_label("Iteration", &[])
          .set_y_label("Error", &[])
          .lines(&x, y, &[Color("red")]);
        fg.show();
    }

    errors
}
