use rulinalg::vector::Vector;

pub struct FunctionParameters {
    vector: Vector<f64>
}

impl FunctionParameters {
    pub fn new(vector: Vector<f64>) -> FunctionParameters {
        FunctionParameters { vector }
    }

    pub fn vector(&self) -> &Vector<f64> {
        &self.vector
    }

    pub fn set_vector(&mut self, new_vector: Vector<f64>) {
        if new_vector.size() != self.vector.size() {
            panic!("FunctionParameters: trying to update function parameters with parameters of different size ({} instead of {}).", new_vector.size(), self.vector.size())
        }

        self.vector = new_vector;
    }
}

#[cfg(test)]
mod tests {
    use super::FunctionParameters;

    #[test]
    fn set_vector() {
        let mut parameters = FunctionParameters::new(vector!(1.0, 2.0, 3.0));

        assert_eq!(parameters.vector(), &vector!(1.0, 2.0, 3.0));
        parameters.set_vector(vector!(3.0, 2.0, 1.0));
        assert_eq!(parameters.vector(), &vector!(3.0, 2.0, 1.0));
    }

    #[test]
    #[should_panic(expected = "FunctionParameters: trying to update function parameters with parameters of different size (2 instead of 3).")]
    fn set_vector_without_enough_parameters() {
        let mut parameters = FunctionParameters::new(vector!(1.0, 2.0, 3.0));

        assert_eq!(parameters.vector(), &vector!(1.0, 2.0, 3.0));
        parameters.set_vector(vector!(3.0, 2.0));
    }

    #[test]
    #[should_panic(expected = "FunctionParameters: trying to update function parameters with parameters of different size (4 instead of 3).")]
    fn set_vector_with_too_many_parameters() {
        let mut parameters = FunctionParameters::new(vector!(1.0, 2.0, 3.0));

        assert_eq!(parameters.vector(), &vector!(1.0, 2.0, 3.0));
        parameters.set_vector(vector!(3.0, 2.0, 1.0, 0.0));
    }
}
