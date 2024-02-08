use std::mem;
use std::time::Instant;
use rusty_brain::*;

fn main() {
    let start_time = Instant::now();

    // test data
    let mut td_in = Matrix::new(4, 2);
    td_in.content[0][0] = 0.;
    td_in.content[1][0] = 0.;
    td_in.content[2][0] = 1.;
    td_in.content[3][0] = 1.;

    td_in.content[0][1] = 0.;
    td_in.content[1][1] = 1.;
    td_in.content[2][1] = 0.;
    td_in.content[3][1] = 1.;

    let mut td_out = Matrix::new(4, 1);
    td_out.content[0][0] = 0.;
    td_out.content[1][0] = 1.;
    td_out.content[2][0] = 1.;
    td_out.content[3][0] = 0.;

    let mut nn = NeuralNetwork::new(vec![2, 2, 1]);
    let mut nn_g = NeuralNetwork::new(vec![2, 2, 1]);

    nn = nn.nn_rand(0., 1.);

    NeuralNetwork::print_nn(nn.clone());
    println!("------------------------------------------------");
    println!("{}", NeuralNetwork::nn_cost(&mut nn.clone(), &td_in, &td_out));

    for _i in 0..20*1000 {
        NeuralNetwork::nn_fin_diff(&mut nn, &mut nn_g, 1e-1, &td_in, &td_out);
        nn = NeuralNetwork::nn_learn(nn, nn_g.clone(), 1e-1);

        // uncomment for live progress
        //println!("{}", NeuralNetwork::nn_cost(&mut nn, &td_in, &td_out));
    }
    println!("{}", NeuralNetwork::nn_cost(&mut nn, &td_in, &td_out));

    for i in 0..2 {
        for j in 0..2 {
            nn.a_s[0].content[0][0] = i as f32;
            nn.a_s[0].content[0][1] = j as f32;

            nn.nn_forward();

            let y = nn.a_s[nn.clone().nn_get_size() - 1].content[0][0];
            println!("{} ^ {} = {}", i, j, y)
        }
    }
    let end_time = Instant::now();

    let total_time = end_time - start_time;
    println!("Time taken: {} seconds {} milliseconds", total_time.as_secs(), total_time.subsec_millis());
}