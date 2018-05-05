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
    // TODO
}
