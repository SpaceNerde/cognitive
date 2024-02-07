use rusty_brain::*;

fn main() {
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
    println!("{}", NeuralNetwork::nn_cost(nn.clone(), td_in.clone(), td_out.clone()));

    for i in 0..20*1000 {
        (nn, nn_g) = NeuralNetwork::nn_fin_diff(nn, nn_g, 1e-1, td_in.clone(), td_out.clone());
        nn = NeuralNetwork::nn_learn(nn, nn_g.clone(), 1e-1);
    }
    println!("{}", NeuralNetwork::nn_cost(nn.clone(), td_in.clone(), td_out.clone()));

    for i in 0..2 {
        for j in 0..2 {
            nn.a_s[0].content[0][0] = i as f32;
            nn.a_s[0].content[0][1] = j as f32;

            nn = nn.nn_forward();
            let y = nn.a_s[nn.clone().nn_get_size() - 1].content[0][0];
            println!("{} ^ {} = {}", i, j, y)
        }
    }
}