use rulinalg::vector::Vector;

use super::ParametricFunction;

// FIXME: does not seem to converge
pub fn gradient_descent_fit<F>(dataset: &Vec<(Vector<f64>, f64)>,
                               parametric_function: &mut F,
                               compute_error_average: &Fn(&F, &Vec<(Vector<f64>, f64)>) -> f64,
                               compute_error_gradients: &Fn(&F, &Vec<(Vector<f64>, f64)>) -> Vector<f64>,
                               num_iterations: u32,
                               learning_rate: f64) -> Vec<f64>
where F: ParametricFunction {
    let mut errors = vec![];

    for _ in 1..num_iterations {
        let new_parameters = {
            let parameters = parametric_function.parameters();
            let gradients = compute_error_gradients(parametric_function, dataset);

            parameters - (gradients * learning_rate)
        };

        errors.push(compute_error_average(parametric_function, dataset));
        parametric_function.set_parameters(new_parameters);
    }

    errors
}
