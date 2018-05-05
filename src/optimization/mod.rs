mod traits;
pub use self::traits::ParametricFunction;

mod types;
pub use self::types::FunctionParameters;
pub use self::types::LinearFunction;

mod least_squares;
pub use self::least_squares::least_squares_fit;

mod gradient_descent;
pub use self::gradient_descent::gradient_descent_fit;
