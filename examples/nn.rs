use rusty_brain::*;

fn main() {
    let mut nn = NeuralNetwork::new(vec![2, 2, 1]);

    NeuralNetwork::print_nn(nn);
}