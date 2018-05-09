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
    use std::f64;

    use optimization::ParametricFunction;
    use optimization::LinearFunction;

    use super::compute_error;
    use super::compute_error_average;
    use super::compute_error_gradients;

    fn build_test_function() -> LinearFunction {
        let mut function = LinearFunction::new(1);

        // f(x) = x
        function.set_parameters(vector!(0.0, 1.0));

        function
    }

    #[test]
    fn compute_error_0() {
        let function = build_test_function();

        assert_eq!(compute_error(&function, &vector!(-3.0), -3.0), 0.0);
        assert_eq!(compute_error(&function, &vector!(3.0), 3.0), 0.0);
    }

    #[test]
    fn compute_error_1() {
        let function = build_test_function();

        assert_eq!(compute_error(&function, &vector!(-2.0), -3.0), 1.0);
        assert_eq!(compute_error(&function, &vector!(-2.0), -1.0), 1.0);
        assert_eq!(compute_error(&function, &vector!(2.0), 1.0), 1.0);
        assert_eq!(compute_error(&function, &vector!(2.0), 3.0), 1.0);
    }

    #[test]
    fn compute_error_squares() {
        let function = build_test_function();

        assert_eq!(compute_error(&function, &vector!(-2.0), -4.0), 4.0);
        assert_eq!(compute_error(&function, &vector!(-2.0), 0.0), 4.0);
        assert_eq!(compute_error(&function, &vector!(-2.0), -5.0), 9.0);
        assert_eq!(compute_error(&function, &vector!(-2.0), 1.0), 9.0);
        assert_eq!(compute_error(&function, &vector!(-2.0), -6.0), 16.0);
        assert_eq!(compute_error(&function, &vector!(-2.0), 2.0), 16.0);
        assert_eq!(compute_error(&function, &vector!(2.0), 0.0), 4.0);
        assert_eq!(compute_error(&function, &vector!(2.0), 4.0), 4.0);
        assert_eq!(compute_error(&function, &vector!(2.0), -1.0), 9.0);
        assert_eq!(compute_error(&function, &vector!(2.0), 5.0), 9.0);
        assert_eq!(compute_error(&function, &vector!(2.0), -2.0), 16.0);
        assert_eq!(compute_error(&function, &vector!(2.0), 6.0), 16.0);
    }

    #[test]
    fn compute_error_average_0() {
        let function = build_test_function();
        let dataset = vec!((vector!(-0.33), -0.33),
                           (vector!(1.0), 1.0),
                           (vector!(0.0), 0.0),
                           (vector!(4.2), 4.2),
                           (vector!(13.37), 13.37),
                           (vector!(3.14), 3.14),
                           (vector!(1.33), 1.33));

        assert_eq!(compute_error_average(&function, &dataset), 0.0);
    }

    #[test]
    fn compute_error_average_1() {
        let function = build_test_function();
        let dataset = vec!((vector!(-0.33), -1.33),
                           (vector!(1.0), 2.0),
                           (vector!(0.0), -1.0),
                           (vector!(4.2), 5.2),
                           (vector!(13.37), 12.37),
                           (vector!(3.14), 2.14),
                           (vector!(1.33), 0.33));

        assert_eq!(compute_error_average(&function, &dataset), 1.0);
    }

    #[test]
    fn compute_error_average_of_squares() {
        let function = build_test_function();
        let dataset = vec!((vector!(0.0), -2.0),
                           (vector!(1.0), 4.0),
                           (vector!(0.0), -4.0),
                           (vector!(4.0), 6.0),
                           (vector!(13.0), 16.0),
                           (vector!(3.0), 7.0),
                           (vector!(1.0), 0.0));

        assert_eq!(compute_error_average(&function, &dataset), (4.0 + 9.0 + 16.0 + 4.0 + 9.0 + 16.0 + 1.0) / 7.0);
    }

    #[test]
    fn compute_error_gradients_with_error_average_0() {
        let function = build_test_function();
        let dataset = vec!((vector!(-0.33), -0.33),
                           (vector!(1.0), 1.0),
                           (vector!(0.0), 0.0),
                           (vector!(4.2), 4.2),
                           (vector!(13.37), 13.37),
                           (vector!(3.14), 3.14),
                           (vector!(1.33), 1.33));

        assert_eq!(compute_error_gradients(&function, &dataset), vector!(0.0, 0.0));
    }

    #[test]
    fn compute_error_gradient_with_error_average_1() {
        let function = build_test_function();
        let dataset = vec!((vector!(-0.33), -1.33),
                           (vector!(1.0), 2.0),
                           (vector!(0.0), -1.0),
                           (vector!(4.2), 5.2),
                           (vector!(13.37), 12.37),
                           (vector!(3.14), 2.14),
                           (vector!(1.33), 0.33));
        let gradients = compute_error_gradients(&function, &dataset).into_vec();

        assert_eq!(2, gradients.len());
        assert_relative_eq!(gradients.as_slice()[0], -2.0, epsilon = f64::EPSILON);
        assert_relative_eq!(
            gradients.as_slice()[1],
            -(2.0 / 7.0) * (-0.33 + 1.0 + 0.0 + 4.2 + 13.37 + 3.14 + 1.33),
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn compute_error_gradients_with_error_squares() {
        let function = build_test_function();
        let dataset = vec!((vector!(0.0), -2.0),
                           (vector!(1.0), 4.0),
                           (vector!(0.0), -4.0),
                           (vector!(4.0), 6.0),
                           (vector!(13.0), 16.0),
                           (vector!(3.0), 7.0),
                           (vector!(1.0), 0.0));
        let gradients = compute_error_gradients(&function, &dataset).into_vec();

        assert_eq!(2, gradients.len());
        assert_relative_eq!(
            gradients.as_slice()[0],
            -(2.0 / 7.0) * (4.0 + 9.0 + 16.0 + 4.0 + 9.0 + 16.0 + 1.0),
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            gradients.as_slice()[1],
            -(2.0 / 7.0) * (4.0 * 0.0 + 9.0 * 1.0 + 16.0 * 0.0 + 4.0 * 4.0 + 9.0 * 13.0 + 16.0 * 3.0 + 1.0 * 1.0),
            epsilon = f64::EPSILON
        );
    }
}
