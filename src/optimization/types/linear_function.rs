use rulinalg::vector::Vector;

use optimization::traits::ParametricFunction;
use super::FunctionParameters;

pub struct LinearFunction {
    input_size: usize,
    parameters: FunctionParameters
}

impl LinearFunction {
    fn add_y_intercept(input: &Vector<f64>) -> Vector<f64> {
        let mut input_with_intercept = input.data().clone();

        input_with_intercept.insert(0, 1.0);

        Vector::new(input_with_intercept)
    }
}

impl ParametricFunction for LinearFunction {
    fn new(input_size: usize) -> Self {
        LinearFunction {
            input_size,
            parameters: FunctionParameters::new(vector![0.0; input_size + 1])
        }
    }

    fn parameters(&self) -> &Vector<f64> {
        self.parameters.vector()
    }

    fn set_parameters(&mut self, new_parameters: Vector<f64>) {
        self.parameters.set_vector(new_parameters)
    }

    fn f(&self, input: &Vector<f64>) -> f64 {
        if input.size() != self.input_size {
            panic!("LinearFunction: trying to apply f() with the wrong number of input variables ({} instead of {}).", input.size(), self.input_size)
        }

        self.parameters.vector().dot(&Self::add_y_intercept(input))
    }

    fn df(&self, input: &Vector<f64>) -> f64 {
        if input.size() != self.input_size {
            panic!("LinearFunction: trying to apply df() with the wrong number of input variables ({} instead of {}).", input.size(), self.input_size)
        }

        let parameters_vector = self.parameters.vector();

        parameters_vector.sum() - parameters_vector[0]
    }

    fn parameter_gradients(&self, input: &Vector<f64>) -> Vector<f64> {
        if input.size() != self.input_size {
            panic!("LinearFunction: trying to get parameter_gradients() with the wrong number of input variables ({} instead of {}).", input.size(), self.input_size)
        }

        Self::add_y_intercept(input)
    }
}

#[cfg(test)]
mod tests {
    use rulinalg::vector::Vector;

    use super::LinearFunction;
    use optimization::traits::ParametricFunction;

    fn test_function(function: &LinearFunction, input: Vector<f64>, y: f64, dx: f64, parameter_gradients: Vector<f64>) {
        assert_eq!(function.f(&input), y);
        assert_eq!(function.df(&input), dx);
        assert_eq!(function.parameter_gradients(&input), parameter_gradients);
    }

    #[test]
    fn add_y_intercept_adds_constant_1_in_front_of_input_vector() {
        assert_eq!(LinearFunction::add_y_intercept(&vector!()), vector!(1.0));
        assert_eq!(LinearFunction::add_y_intercept(&vector!(1.0)), vector!(1.0, 1.0));
        assert_eq!(LinearFunction::add_y_intercept(&vector!(2.0)), vector!(1.0, 2.0));
        assert_eq!(LinearFunction::add_y_intercept(&vector!(3.0)), vector!(1.0, 3.0));
        assert_eq!(LinearFunction::add_y_intercept(&vector!(3.0, 1.0, 2.0)), vector!(1.0, 3.0, 1.0, 2.0));
    }

    #[test]
    fn parameters_update() {
        let mut function = LinearFunction::new(3);

        assert_eq!(*function.parameters(), vector!(0.0, 0.0, 0.0, 0.0));
        function.set_parameters(vector!(0.0, 3.0, 2.0, 1.0));
        assert_eq!(*function.parameters(), vector!(0.0, 3.0, 2.0, 1.0));
    }

    #[test]
    #[should_panic(expected = "LinearFunction: trying to apply f() with the wrong number of input variables (1 instead of 2).")]
    fn f_without_enough_input_variables() {
        let function = LinearFunction::new(2);

        function.f(&vector!(1.0));
    }

    #[test]
    #[should_panic(expected = "LinearFunction: trying to apply f() with the wrong number of input variables (3 instead of 2).")]
    fn f_with_too_many_input_variables() {
        let function = LinearFunction::new(2);

        function.f(&vector!(1.0, 2.0, 3.0));
    }

    #[test]
    #[should_panic(expected = "LinearFunction: trying to apply df() with the wrong number of input variables (1 instead of 2).")]
    fn df_without_enough_input_variables() {
        let function = LinearFunction::new(2);

        function.df(&vector!(1.0));
    }

    #[test]
    #[should_panic(expected = "LinearFunction: trying to apply df() with the wrong number of input variables (3 instead of 2).")]
    fn df_with_too_many_input_variables() {
        let function = LinearFunction::new(2);

        function.df(&vector!(1.0, 2.0, 3.0));
    }

    #[test]
    #[should_panic(expected = "LinearFunction: trying to get parameter_gradients() with the wrong number of input variables (1 instead of 2).")]
    fn parameter_gradients_without_enough_input_variables() {
        let function = LinearFunction::new(2);

        function.parameter_gradients(&vector!(1.0));
    }

    #[test]
    #[should_panic(expected = "LinearFunction: trying to get parameter_gradients() with the wrong number of input variables (3 instead of 2).")]
    fn parameter_gradients_with_too_many_input_variables() {
        let function = LinearFunction::new(2);

        function.parameter_gradients(&vector!(1.0, 2.0, 3.0));
    }

    #[test]
    fn function_without_input_variables() {
        let mut function = LinearFunction::new(0);

        //  f(x) = 3
        // df(x) = 0
        function.set_parameters(vector!(3.0));

        test_function(&function, vector!(), 3.0, 0.0, vector!(1.0))
    }

    #[test]
    fn function_without_y_intercept() {
        let mut function = LinearFunction::new(1);

        //  f(x) = 1x
        // df(x) = 1
        function.set_parameters(vector!(0.0, 1.0));

        test_function(&function, vector!(0.0), 0.0, 1.0, vector!(1.0, 0.0));
        test_function(&function, vector!(1.0), 1.0, 1.0, vector!(1.0, 1.0));
        test_function(&function, vector!(2.0), 2.0, 1.0, vector!(1.0, 2.0));
        test_function(&function, vector!(3.0), 3.0, 1.0, vector!(1.0, 3.0));
    }

    #[test]
    fn function_with_parameter_set_to_0() {
        let mut function0 = LinearFunction::new(1);
        let mut function1 = LinearFunction::new(1);

        //  f(x) = 0 + 0x
        // df(x) = 0
        function0.set_parameters(vector!(0.0, 0.0));

        test_function(&function0, vector!(0.0), 0.0, 0.0, vector!(1.0, 0.0));
        test_function(&function0, vector!(1.0), 0.0, 0.0, vector!(1.0, 1.0));
        test_function(&function0, vector!(2.0), 0.0, 0.0, vector!(1.0, 2.0));
        test_function(&function0, vector!(3.0), 0.0, 0.0, vector!(1.0, 3.0));

        //  f(x) = 1 + 0x
        // df(x) = 0
        function1.set_parameters(vector!(1.0, 0.0));

        test_function(&function1, vector!(0.0), 1.0, 0.0, vector!(1.0, 0.0));
        test_function(&function1, vector!(1.0), 1.0, 0.0, vector!(1.0, 1.0));
        test_function(&function1, vector!(2.0), 1.0, 0.0, vector!(1.0, 2.0));
        test_function(&function1, vector!(3.0), 1.0, 0.0, vector!(1.0, 3.0));
    }

    #[test]
    fn function_with_one_input_variable() {
        let mut function = LinearFunction::new(1);

        //  f(x) = 1 + 2x
        // df(x) = 2
        function.set_parameters(vector!(1.0, 2.0));

        test_function(&function, vector!(0.0), 1.0, 2.0, vector!(1.0, 0.0));
        test_function(&function, vector!(1.0), 3.0, 2.0, vector!(1.0, 1.0));
        test_function(&function, vector!(2.0), 5.0, 2.0, vector!(1.0, 2.0));
        test_function(&function, vector!(3.0), 7.0, 2.0, vector!(1.0, 3.0));
    }

    #[test]
    fn function_with_two_input_variables() {
        let mut function = LinearFunction::new(2);

        //  f(x, y) = 1 + 2x + y
        // df(x, y) = 3
        function.set_parameters(vector!(1.0, 2.0, 1.0));

        test_function(&function, vector!(0.0, 3.0), 4.0, 3.0, vector!(1.0, 0.0, 3.0));
        test_function(&function, vector!(1.0, 2.0), 5.0, 3.0, vector!(1.0, 1.0, 2.0));
        test_function(&function, vector!(2.0, 1.0), 6.0, 3.0, vector!(1.0, 2.0, 1.0));
        test_function(&function, vector!(3.0, 0.0), 7.0, 3.0, vector!(1.0, 3.0, 0.0));
    }

    #[test]
    fn function_with_three_input_variables() {
        let mut function = LinearFunction::new(3);

        //  f(x, y, z) = 1 + 3x + 2y + z
        // df(x, y, z) = 6
        function.set_parameters(vector!(1.0, 3.0, 2.0, 1.0));

        test_function(&function, vector!(0.0, 3.0, 1.0), 8.0, 6.0, vector!(1.0, 0.0, 3.0, 1.0));
        test_function(&function, vector!(1.0, 2.0, 0.0), 8.0, 6.0, vector!(1.0, 1.0, 2.0, 0.0));
        test_function(&function, vector!(2.0, 1.0, 3.0), 12.0, 6.0, vector!(1.0, 2.0, 1.0, 3.0));
        test_function(&function, vector!(3.0, 0.0, 2.0), 12.0, 6.0, vector!(1.0, 3.0, 0.0, 2.0));
    }
}
