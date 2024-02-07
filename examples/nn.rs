use rusty_brain::*;

fn main() {
    let mut nn = NeuralNetwork::new(vec![2, 2, 1]);

    nn = nn.nn_rand(0., 1.);
    NeuralNetwork::print_nn(nn);
}