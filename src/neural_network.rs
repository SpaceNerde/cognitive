// credit Tsoding Daily

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

    pub fn nn_forward(mut self) -> Self {
        for i in 0..self.clone().nn_get_size() - 1 {
            self.a_s[i + 1] = Matrix::matrix_dot(self.a_s[i].clone(), self.w_s[i].clone());
            self.a_s[i + 1] = Matrix::matrix_sum(self.a_s[i + 1].clone(), self.b_s[i].clone());
            self.a_s[i + 1] = Matrix::matrix_sigmoid(self.a_s[i + 1].clone());
        }

        self
    }

    // i call it THE BLOCK
    // readability -1000000
    pub fn nn_fin_diff(mut neural_network: NeuralNetwork, mut g: NeuralNetwork, modi: f32, t_in: Matrix, t_out: Matrix) -> (NeuralNetwork, NeuralNetwork){
        let mut buffer;
        let c = NeuralNetwork::nn_cost(neural_network.clone(), t_in.clone(), t_out.clone());

        for i in 0..neural_network.clone().nn_get_size() - 1 {
            for j in 0..neural_network.w_s[i].rows {
                for k in 0..neural_network.w_s[i].cols {
                    buffer = neural_network.w_s[i].content[j as usize][k as usize];
                    neural_network.w_s[i].content[j as usize][k as usize] += modi;
                    g.w_s[i].content[j as usize][k as usize] = (NeuralNetwork::nn_cost(neural_network.clone(), t_in.clone(), t_out.clone()) - c)/modi;
                    neural_network.w_s[i].content[j as usize][k as usize] = buffer;
                }
            }

            for j in 0..neural_network.b_s[i].rows {
                for k in 0..neural_network.b_s[i].cols {
                    buffer = neural_network.b_s[i].content[j as usize][k as usize];
                    neural_network.b_s[i].content[j as usize][k as usize] += modi;
                    g.b_s[i].content[j as usize][k as usize] = (NeuralNetwork::nn_cost(neural_network.clone(), t_in.clone(), t_out.clone()) - c)/modi;
                    neural_network.b_s[i].content[j as usize][k as usize] = buffer;
                }
            }
        }

        (neural_network, g)
    }

    pub fn nn_cost(mut neural_network: NeuralNetwork, t_in: Matrix, t_out: Matrix) -> f32{
        assert_eq!(t_in.rows, t_out.rows);
        assert_eq!(t_out.cols, neural_network.a_s[neural_network.clone().nn_get_size() - 1].cols);

        // actual cost
        let mut c = 0.;

        // sizes
        let n0 = t_in.rows;
        let n1 = t_out.cols;

        for i in 0..n0 {
            let mut x = Matrix::matrix_row(t_in.clone(), i);
            let mut y = Matrix::matrix_row(t_out.clone(), i);

            neural_network.a_s[0] = Matrix::matrix_copy(neural_network.a_s[0].clone(), x.clone());
            neural_network = self::NeuralNetwork::nn_forward(neural_network);

            for j in 0..n1 {
                let mut diff = neural_network.a_s[neural_network.clone().nn_get_size() - 1].content[0][j as usize] - y.content[0][j as usize];
                c += diff.powf(2.);
            }
        }

        c/n0 as f32
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

    pub fn nn_get_size(self) -> usize {
        self.architecture.len()
    }
}

