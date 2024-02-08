// credit Tsoding Daily

// {2, 3, 1}
// 2 inputs
// 3 neurons in one hidden layer
// 1 output

use crate::{Matrix, sigmoid};

// {2, 3, 3, 3, 1}
// same thing but 3 hidden layers with 3 neurons each
#[derive(Debug, Clone, Default)]
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
        println!("Neural Network: ");
        for i in 0..neural_network.clone().nn_get_size() - 1 {
            print!("  weight {i}: ");
            neural_network.w_s[i].clone().print_matrix();
            print!("  bias {i}: ");
            neural_network.b_s[i].clone().print_matrix();
        }
    }

    pub fn nn_rand(mut self, min: f32, max: f32) -> Self{
        for i in 0..self.clone().nn_get_size() - 1 {
            self.w_s[i] = self.w_s[i].clone().matrix_rand(min, max);
            self.b_s[i] = self.b_s[i].clone().matrix_rand(min, max);
        }

        self
    }

    pub fn nn_forward(&mut self) {
        for i in 0..self.nn_get_size() - 1 {
            let dot_result = Matrix::matrix_dot(&self.a_s[i], &self.w_s[i]);
            let mut sum_result = Matrix::matrix_sum(&dot_result, &self.b_s[i]);
            sum_result.map(|x| sigmoid(x));  // Apply sigmoid to each element
            self.a_s[i + 1] = sum_result;
        }
    }

    // i call it THE BLOCK
    // readability -1000000
    pub fn nn_fin_diff<'a>(
        neural_network: &'a mut NeuralNetwork,
        nn_g: &'a mut NeuralNetwork,
        modi: f32,
        t_in: &Matrix,
        t_out: &Matrix,
    ) {
        let mut buffer;
        let c = NeuralNetwork::nn_cost(neural_network, t_in, t_out);
        for i in 0..neural_network.nn_get_size() - 1 {
            for j in 0..neural_network.w_s[i].rows {
                for k in 0..neural_network.w_s[i].cols {
                    buffer = neural_network.w_s[i].content[j as usize][k as usize];
                    neural_network.w_s[i].content[j as usize][k as usize] += modi;
                    nn_g.w_s[i].content[j as usize][k as usize] = (NeuralNetwork::nn_cost(neural_network, t_in, t_out) - c) / modi;
                    neural_network.w_s[i].content[j as usize][k as usize] = buffer;
                }
            }

            for j in 0..neural_network.b_s[i].rows {
                for k in 0..neural_network.b_s[i].cols {
                    buffer = neural_network.b_s[i].content[j as usize][k as usize];
                    neural_network.b_s[i].content[j as usize][k as usize] += modi;
                    nn_g.b_s[i].content[j as usize][k as usize] = (NeuralNetwork::nn_cost(neural_network, t_in, t_out) - c) / modi;
                    neural_network.b_s[i].content[j as usize][k as usize] = buffer;
                }
            }
        }
    }

    pub fn nn_cost(neural_network: &mut NeuralNetwork, t_in: &Matrix, t_out: &Matrix) -> f32 {
        assert_eq!(t_in.rows, t_out.rows);
        assert_eq!(
            t_out.cols,
            neural_network.a_s[neural_network.nn_get_size() - 1].cols
        );

        // Actual cost
        let mut c = 0.;

        // Sizes
        let n0 = t_in.rows;
        let n1 = t_out.cols;

        for i in 0..n0 {
            let x = Matrix::matrix_row(t_in, i);
            let y = Matrix::matrix_row(t_out, i);

            neural_network.a_s[0] = x;
            neural_network.nn_forward();

            for j in 0..n1 {
                let diff = neural_network.a_s[neural_network.nn_get_size() - 1].content[0][j as usize] - y.content[0][j as usize];
                c += diff.powf(2.);
            }
        }

        c / n0 as f32
    }

    pub fn nn_learn(mut neural_network: NeuralNetwork, nn_g: NeuralNetwork, learning_rate: f32) -> NeuralNetwork {
        for i in 0..neural_network.clone().nn_get_size() - 1 {
            for j in 0..neural_network.w_s[i].rows {
                for k in 0..neural_network.w_s[i].cols {
                    neural_network.w_s[i].content[j as usize][k as usize] -= learning_rate * nn_g.w_s[i].content[j as usize][k as usize];
                }
            }

            for j in 0..neural_network.b_s[i].rows {
                for k in 0..neural_network.b_s[i].cols {
                    neural_network.b_s[i].content[j as usize][k as usize] -= learning_rate * nn_g.b_s[i].content[j as usize][k as usize];
                }
            }
        }

        neural_network
    }

    pub fn nn_get_size(&self) -> usize {
        self.architecture.len()
    }
}

