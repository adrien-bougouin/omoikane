use rulinalg::vector::Vector;

use super::ParametricFunction;
use super::gradient_descent_fit;

fn compute_error<F>(function: &F, input: &Vector<f64>, y: f64) -> f64
where F: ParametricFunction {
    (y - function.f(input)).powi(2)
}

fn compute_error_average<F>(function: &F, dataset: &Vec<(Vector<f64>, f64)>) -> f64
where F: ParametricFunction {
    let n = dataset.len() as f64;
    let mut squared_errors_sum = 0.0;
    let data_iterator = dataset.iter();

    for ref labeled_data in data_iterator {
        squared_errors_sum += compute_error(function, &labeled_data.0, labeled_data.1);
    }

    squared_errors_sum / n
}

fn compute_error_gradients<F>(function: &F, dataset: &Vec<(Vector<f64>, f64)>) -> Vector<f64>
where F: ParametricFunction {
    let n = dataset.len() as f64;
    let mut gradients = vec![0.0; function.parameters().size()];
    let data_iterator = dataset.iter();

    for ref labeled_data in data_iterator {
        let previous_gradients = gradients;
        let ref input = labeled_data.0;
        let label = labeled_data.1;
        let error = compute_error(function, input, label);
        let parameter_gradients = function.parameter_gradients(input);
        let gradients_zip_iterator = previous_gradients.iter().zip(parameter_gradients.iter());

        gradients = vec!();
        for (gradient, parameter_gradient) in gradients_zip_iterator {
            gradients.push(gradient + (-2.0 / n) * (error * parameter_gradient))
        }
    }

    Vector::new(gradients)
}

pub fn least_squares_fit<F>(function: &mut F,
                            dataset: &Vec<(Vector<f64>, f64)>,
                            num_iterations: u32,
                            learning_rate: f64) -> Vec<f64>
where F: ParametricFunction {
    gradient_descent_fit(dataset,
                         function,
                         &compute_error_average,
                         &compute_error_gradients,
                         num_iterations,
                         learning_rate)
}

#[cfg(test)]
mod tests {
    // TODO: move to integration tests???
    mod linear_function {
        use optimization::least_squares::compute_error;
        use optimization::ParametricFunction;
        use optimization::LinearFunction;

        #[test]
        fn compute_error_0() {
            let mut function = LinearFunction::new(1);

            // f(x) = 1 + 2x
            function.set_parameters(vector!(1.0, 2.0));

            // f(2) = 5
            assert_eq!(compute_error(&function, &vector!(2.0), 5.0), 0.0);
        }

        #[test]
        fn compute_error_1() {
            let mut function = LinearFunction::new(1);

            // f(x) = 1 + 2x
            function.set_parameters(vector!(1.0, 2.0));

            // f(2) = 5
            assert_eq!(compute_error(&function, &vector!(2.0), 4.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(2.0), 6.0), 1.0);
        }

        #[test]
        fn compute_error_squares() {
            let mut function = LinearFunction::new(1);

            // f(x) = 1 + 2x
            function.set_parameters(vector!(1.0, 2.0));

            // f(2) = 5
            assert_eq!(compute_error(&function, &vector!(2.0), 3.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(2.0), 7.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(2.0), 2.0), 9.0);
            assert_eq!(compute_error(&function, &vector!(2.0), 8.0), 9.0);
            assert_eq!(compute_error(&function, &vector!(2.0), 1.0), 16.0);
            assert_eq!(compute_error(&function, &vector!(2.0), 9.0), 16.0);
        }

        #[test]
        fn compute_error_with_fx_equal_0() {
            let mut function = LinearFunction::new(1);

            // f(x) = 0
            function.set_parameters(vector!(0.0, 0.0));

            assert_eq!(compute_error(&function, &vector!(-3.0), 2.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(-2.0), 1.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(-1.0), 0.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(0.0), 0.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(0.0), 1.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(0.0), 2.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(1.0), 0.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(2.0), 1.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(3.0), 2.0), 4.0);
        }

        #[test]
        fn compute_error_with_fx_equal_minus_1() {
            let mut function = LinearFunction::new(1);

            // f(x) = 1
            function.set_parameters(vector!(-1.0, 0.0));

            assert_eq!(compute_error(&function, &vector!(-3.0), -3.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(-2.0), -2.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(-1.0), -1.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(0.0), -1.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(0.0), -2.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(0.0), -3.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(1.0), -1.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(2.0), -2.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(3.0), -3.0), 4.0);
        }

        #[test]
        fn compute_error_with_fx_equal_1() {
            let mut function = LinearFunction::new(1);

            // f(x) = 1
            function.set_parameters(vector!(1.0, 0.0));

            assert_eq!(compute_error(&function, &vector!(-3.0), 3.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(-2.0), 2.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(-1.0), 1.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(0.0), 1.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(0.0), 2.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(0.0), 3.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(1.0), 1.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(2.0), 2.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(3.0), 3.0), 4.0);
        }

        #[test]
        fn compute_error_with_fx_equal_x() {
            let mut function = LinearFunction::new(1);

            // f(x) = x
            function.set_parameters(vector!(0.0, 1.0));

            assert_eq!(compute_error(&function, &vector!(-3.0), -5.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(-3.0), -1.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(-2.0), -3.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(-2.0), -1.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(-1.0), -1.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(0.0), 0.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(0.0), -1.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(0.0), 1.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(0.0), -2.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(0.0), 2.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(1.0), 1.0), 0.0);
            assert_eq!(compute_error(&function, &vector!(2.0), 1.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(2.0), 3.0), 1.0);
            assert_eq!(compute_error(&function, &vector!(3.0), 1.0), 4.0);
            assert_eq!(compute_error(&function, &vector!(3.0), 5.0), 4.0);
        }
    }
}
