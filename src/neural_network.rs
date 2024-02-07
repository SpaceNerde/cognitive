// {2, 3, 1}
// 2 inputs
// 3 neurons in one hidden layer
// 1 output

use crate::Matrix;

// {2, 3, 3, 3, 1}
// same thing but 3 hidden layers with 3 neurons each
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub architecture: Vec<i32>,
    // activations
    pub a_s: Vec<Matrix>,
    // weights
    pub w_s: Vec<Matrix>,
    // biases
    pub b_s: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(arch: Vec<i32>) -> Self{
        assert_ne!(arch, vec![]);

        let size = arch.len();

        let mut w_s = vec![];
        let mut b_s = vec![];
        let mut a_s = vec![];

        a_s.insert(0, Matrix::new(1, arch[0]));
        for i in 1..size {
            w_s.insert(i - 1, Matrix::new(a_s[i - 1].cols, arch[i]));
            b_s.insert(i - 1, Matrix::new(1, arch[i]));
            a_s.insert(i, Matrix::new(1, arch[i]));
        }

        NeuralNetwork {
            architecture: arch,
            a_s,
            w_s,
            b_s,
        }
    }

    pub fn print_nn(neural_network: NeuralNetwork) {
        let size = neural_network.architecture.len();

        println!("Neural Network: ");
        for i in 0..size - 1 {
            print!("  weight {i}: ");
            neural_network.w_s[i].clone().print_matrix();
            print!("  bias {i}: ");
            neural_network.b_s[i].clone().print_matrix();
        }
    }

    pub fn nn_rand(mut self, min: f32, max: f32) -> Self{
        let size = self.architecture.len();

        for i in 0..size - 1 {
            self.w_s[i] = self.w_s[i].clone().matrix_rand(min, max);
            self.b_s[i] = self.b_s[i].clone().matrix_rand(min, max);
        }

        self
    }
}

