use rulinalg::vector::Vector;

use Model;
use optimization::ParametricFunction;
use optimization::LinearFunction;
use optimization::least_squares_fit;

pub struct LinearRegressionModel {
    learning_rate: f64,
    max_iterations: u32,
    linear_function: Option<LinearFunction>
}

impl LinearRegressionModel {
        pub fn new(learning_rate: f64, max_iterations: u32) -> LinearRegressionModel {
        LinearRegressionModel {
            learning_rate,
            max_iterations,
            linear_function: None
        }
    }
}

impl Model<Vector<f64>, f64> for LinearRegressionModel {
    // TODO: create Dataset type to ensure that all data have the same number of features
    // TODO: move Dataset type to Generic Type for single fit function with either supervised or
    // unsupervised??? (doing so will loose the availability to support both)
    fn fit_supervised_dataset(&mut self, dataset: &Vec<(Vector<f64>, f64)>) {
        let input_size = match dataset.first() {
            None => 0,
            Some(ref value) => value.0.size()
        };

        if input_size > 0 {
            let mut function = LinearFunction::new(input_size);

            least_squares_fit(&mut function, dataset, self.learning_rate, self.max_iterations);

            self.linear_function = Some(function);
        }
    }

    fn predict(&self, data: &Vector<f64>) -> f64 {
        match self.linear_function {
            None => panic!("LinearRegressionModel: trying to predict before fitting."),
            Some(ref function) => function.f(data)
        }
    }
}
