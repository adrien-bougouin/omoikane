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
    let errors_sum = dataset.iter()
                            .map(|&(ref x, y)| compute_error(function, x, y))
                            .sum::<f64>();

    errors_sum / n
}

fn compute_error_gradients<F>(function: &F, dataset: &Vec<(Vector<f64>, f64)>) -> Vector<f64>
where F: ParametricFunction {
    let n = dataset.len() as f64;
    let mut gradients = vec![0.0; function.parameters().size()];

    for &(ref x, y) in dataset.iter() {
        let previous_gradients = gradients;
        let error = compute_error(function, x, y);
        let parameter_gradients = function.parameter_gradients(x);

        gradients = previous_gradients.iter()
                                      .zip(parameter_gradients.iter())
                                      .map(|(acc, g)| acc + (-2.0 / n) * (error * g))
                                      .collect()
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
                           (vector!(13.36), 13.36),
                           (vector!(3.13), 3.13),
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
                           (vector!(13.36), 12.36),
                           (vector!(3.13), 2.13),
                           (vector!(1.33), 0.33));

        assert_eq!(compute_error_average(&function, &dataset), 1.0);
    }

    #[test]
    fn compute_error_average_of_squares() {
        let function = build_test_function();
        let dataset = vec!((vector!(-0.33), -2.33),
                           (vector!(1.0), 4.0),
                           (vector!(0.0), -4.0),
                           (vector!(4.2), 6.2),
                           (vector!(13.36), 16.36),
                           (vector!(3.13), 7.13),
                           (vector!(1.33), 0.33));

        assert_relative_eq!(compute_error_average(&function, &dataset),
                            (4.0 + 9.0 + 16.0 + 4.0 + 9.0 + 16.0 + 1.0) / 7.0);
    }

    #[test]
    fn compute_error_gradients_with_error_average_0() {
        let function = build_test_function();
        let dataset = vec!((vector!(-0.33), -0.33),
                           (vector!(1.0), 1.0),
                           (vector!(0.0), 0.0),
                           (vector!(4.2), 4.2),
                           (vector!(13.36), 13.36),
                           (vector!(3.13), 3.13),
                           (vector!(1.33), 1.33));

        assert_eq!(compute_error_gradients(&function, &dataset),
                   vector!(0.0, 0.0));
    }

    #[test]
    fn compute_error_gradient_with_error_average_1() {
        let function = build_test_function();
        let dataset = vec!((vector!(-0.33), -1.33),
                           (vector!(1.0), 2.0),
                           (vector!(0.0), -1.0),
                           (vector!(4.2), 5.2),
                           (vector!(13.36), 12.36),
                           (vector!(3.13), 2.13),
                           (vector!(1.33), 0.33));
        let gradients = compute_error_gradients(&function, &dataset).into_vec();

        assert_eq!(2, gradients.len());
        assert_relative_eq!(gradients.as_slice()[0], -2.0, epsilon = f64::EPSILON);
        assert_relative_eq!(gradients.as_slice()[1],
                            -(2.0 / 7.0) * (-0.33 + 1.0 + 0.0 + 4.2 + 13.36 + 3.13 + 1.33),
                            epsilon = f64::EPSILON);
    }

    #[test]
    fn compute_error_gradients_with_error_squares() {
        let function = build_test_function();
        let dataset = vec!((vector!(-0.33), -2.33),
                           (vector!(1.0), 4.0),
                           (vector!(0.0), -4.0),
                           (vector!(4.2), 6.2),
                           (vector!(13.36), 16.36),
                           (vector!(3.13), 7.13),
                           (vector!(1.33), 0.33));
        let gradients = compute_error_gradients(&function, &dataset).into_vec();

        assert_eq!(2, gradients.len());
        assert_relative_eq!(gradients.as_slice()[0],
                            -(2.0 / 7.0) * (4.0 + 9.0 + 16.0 + 4.0 + 9.0 + 16.0 + 1.0),
                            epsilon = f64::EPSILON);
        assert_relative_eq!(gradients.as_slice()[1],
                            -(2.0 / 7.0) * (4.0 * -0.33 + 9.0 * 1.0 + 16.0 * 0.0 + 4.0 * 4.2 + 9.0 * 13.36 + 16.0 * 3.13 + 1.0 * 1.33),
                            epsilon = f64::EPSILON);
    }
}
