// https://www.youtube.com/watch?v=L1TbWe8bVOc
// credit: Tsoding Daily

use std::mem::size_of;
use rusty_brain::Matrix;

#[derive(Debug, Clone)]
struct Xor{
    // input
    a0: Matrix,

    // weights
    w1: Matrix,
    w2: Matrix,

    // biases
    b1: Matrix,
    b2: Matrix,

    // buffer
    a1: Matrix,
    a2: Matrix,
}

impl Xor {
    fn new() -> Self{
        // define standard values
        let a0 = Matrix::new(1, 2);
        let w1 = Matrix::new(2, 2);
        let w2 = Matrix::new(2, 1);
        let b1 = Matrix::new(1, 2);
        let b2 = Matrix::new(1, 1);
        let a1 = Matrix::new(1,2);
        let a2 = Matrix::new(1,1);

        Xor {
            a0,
            w1,
            w2,
            b1,
            b2,
            a1,
            a2,
        }
    }
}

fn cost(mut m: Xor, td_in: Matrix, td_out: Matrix) -> f32{
    assert_eq!(td_in.rows, td_out.rows);
    assert_eq!(td_out.cols, m.a2.cols);

    // actual cost
    let mut c = 0.;

    // sizes
    let n0 = td_in.rows;
    let n1 = td_out.cols;

    for i in 0..n0 {
        let mut x = Matrix::matrix_row(td_in.clone(), i);
        let mut y = Matrix::matrix_row(td_out.clone(), i);

        m.a0 = Matrix::matrix_copy(m.a0, x.clone());
        m = forward_xor(m);

        for j in 0..n1 {
            let mut diff = m.a2.content[0][j as usize] - y.content[0][j as usize];
            c += diff.powf(2.);
        }
    }

    c/n0 as f32
}

fn forward_xor(mut m: Xor) -> Xor{
    m.a1 = Matrix::matrix_dot(m.a0.clone(), m.w1.clone());
    m.a1 = Matrix::matrix_sum(m.b1.clone(), m.a1.clone());
    m.a1 = Matrix::matrix_sigmoid(m.a1);

    m.a2 = Matrix::matrix_dot(m.a1.clone(), m.w2.clone());
    m.a2 = Matrix::matrix_sum(m.a2.clone(), m.b2.clone());
    m.a2 = Matrix::matrix_sigmoid(m.a2);

    m
}

fn finite_diff(mut m: Xor, mut g: Xor, modi: f32, td_in: Matrix, td_out: Matrix) -> (Xor, Xor, Matrix, Matrix){
    let mut buffer;

    let mut c = cost(m.clone(), td_in.clone(), td_out.clone());

    // I never steal code from Tsoding Daily again.
    for i in 0..m.w1.rows {
        for j in 0..m.w1.cols {
            buffer = m.w1.content[i as usize][j as usize];
            m.w1.content[i as usize][j as usize] += modi;
            g.w1.content[i as usize][j as usize] = (cost(m.clone(), td_in.clone(), td_out.clone()) - c)/modi;
            m.w1.content[i as usize][j as usize] = buffer;
        }
    }

    for i in 0..m.b1.rows {
        for j in 0..m.b1.cols {
            buffer = m.b1.content[i as usize][j as usize];
            m.b1.content[i as usize][j as usize] += modi;
            g.b1.content[i as usize][j as usize] = (cost(m.clone(), td_in.clone(), td_out.clone()) - c)/modi;
            m.b1.content[i as usize][j as usize] = buffer;
        }
    }

    for i in 0..m.w2.rows {
        for j in 0..m.w2.cols {
            buffer = m.w2.content[i as usize][j as usize];
            m.w2.content[i as usize][j as usize] += modi;
            g.w2.content[i as usize][j as usize] = (cost(m.clone(), td_in.clone(), td_out.clone()) - c)/modi;
            m.w2.content[i as usize][j as usize] = buffer;
        }
    }

    for i in 0..m.b2.rows {
        for j in 0..m.b2.cols {
            buffer = m.b2.content[i as usize][j as usize];
            m.b2.content[i as usize][j as usize] += modi;
            g.b2.content[i as usize][j as usize] = (cost(m.clone(), td_in.clone(), td_out.clone()) - c)/modi;
            m.b2.content[i as usize][j as usize] = buffer;
        }
    }

    (m, g, td_in, td_out)
}

// ok yeah u do u Tsoding Daily
fn learn(mut m: Xor, mut g: Xor, rate: f32) -> (Xor, Xor) {
    for i in 0..m.w1.rows {
        for j in 0..m.w1.cols {
            m.w1.content[i as usize][j as usize] -= rate*(g.w1.content[i as usize][j as usize]);
        }
    }

    for i in 0..m.b1.rows {
        for j in 0..m.b1.cols {
            m.b1.content[i as usize][j as usize] -= rate*(g.b1.content[i as usize][j as usize]);
        }
    }

    for i in 0..m.w2.rows {
        for j in 0..m.w2.cols {
            m.w2.content[i as usize][j as usize] -= rate*(g.w2.content[i as usize][j as usize]);
        }
    }

    for i in 0..m.b2.rows {
        for j in 0..m.b2.cols {
            m.b2.content[i as usize][j as usize] -= rate*(g.b2.content[i as usize][j as usize]);
        }
    }
    (m, g)
}

fn main() {
    // training data
    let training_data =[
        [0, 0, 0,],
        [0, 1, 1,],
        [1, 0, 1,],
        [1, 1, 0,],
    ];

    let n = training_data.len()/(training_data[0].len()/3);

    // hey why dont u just use an for loop?
    // cause duh?
    // wait who are u and why are u actually reading my comments?
    // this lib is actually shit and u probably should not use it xD
    // ...
    let mut td_in = Matrix::new(n as i32, 2);
    td_in.content[0][0] = 0.;
    td_in.content[1][0] = 0.;
    td_in.content[2][0] = 1.;
    td_in.content[3][0] = 1.;

    td_in.content[0][1] = 0.;
    td_in.content[1][1] = 1.;
    td_in.content[2][1] = 0.;
    td_in.content[3][1] = 1.;

    let mut td_out = Matrix::new(n as i32, 1);
    td_out.content[0][0] = 0.;
    td_out.content[1][0] = 1.;
    td_out.content[2][0] = 1.;
    td_out.content[3][0] = 0.;

    let mut xor = Xor::new();
    let mut xor_g = xor.clone();

    // fill matrix with random numbers to initialize
    xor.w1 = xor.w1.matrix_rand();
    xor.w2 = xor.w2.matrix_rand();
    xor.b1 = xor.b1.matrix_rand();
    xor.b2 = xor.b2.matrix_rand();

    println!("cost: {}", cost(xor.clone(), td_in.clone(), td_out.clone()));

    for i in 0..10000 {
        (xor, xor_g, _, _) = finite_diff(xor, xor_g, 1e-1, td_in.clone(), td_out.clone());
        let (new_xor, new_xor_g) = learn(xor, xor_g, 1e-1);
        xor = new_xor;
        xor_g = new_xor_g;
        // Uncomment the following line if you want to see the cost during training
        println!("cost: {}", cost(xor.clone(), td_in.clone(), td_out.clone()));
    }

    println!("---------------------------------------------------");

    for i in 0..2 {
        for j in 0..2 {
            xor.a0.content[0][0] = i as f32;
            xor.a0.content[0][1] = j as f32;

            xor = forward_xor(xor);
            let y = xor.a2.content[0][0];

            println!("{} ^ {} = {}", i, j, y);
        }
    }
}