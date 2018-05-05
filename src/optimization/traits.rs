use rulinalg::vector::Vector;

pub trait ParametricFunction {
    fn new(input_size: usize) -> Self;
    fn parameters(&self) -> &Vector<f64>;
    fn set_parameters(&mut self, new_parameters: Vector<f64>);
    fn f(&self, input: &Vector<f64>) -> f64;
    fn df(&self, input: &Vector<f64>) -> f64;
    fn parameter_gradients(&self, input: &Vector<f64>) -> Vector<f64>;
}
