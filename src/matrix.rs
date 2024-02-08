use rand::Rng;
use crate::sigmoid;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: i32,
    pub cols: i32,
    pub content: Vec<Vec<f32>>
}

impl Matrix {
    pub fn new(rows: i32, cols: i32) -> Self{
        let content = vec![vec![0.0; cols as usize]; rows as usize];

        Matrix {
            rows,
            cols,
            content,
        }
    }

    pub fn map<F>(&mut self, func: F)
        where
            F: Fn(f32) -> f32,
    {
        for row in &mut self.content {
            for element in row {
                *element = func(*element);
            }
        }
    }

    pub fn print_matrix(&self) {
        println!("  [");
        for i in 0..self.rows {
            for j in 0..self.cols {
                print!("        {:?}", self.content[i as usize][j as usize])
            }
            print!("\n");
        }
        println!("  ]");
    }

    pub fn matrix_dot(matrix_1: &Matrix, matrix_2: &Matrix) -> Matrix{
        assert_eq!(matrix_1.cols, matrix_2.rows, "Matrix dimensions mismatch");

        let mut buffer_matrix = Matrix::new(
            matrix_1.rows,
            matrix_2.cols
        );

        for i in 0..matrix_1.rows {
            for j in 0..matrix_2.cols {
                for k in 0..matrix_1.cols {
                    buffer_matrix.content[i as usize][j as usize] += matrix_1.content[i as usize][k as usize] * matrix_2.content[k as usize][j as usize];
                }
            }
        }

        buffer_matrix
    }

    pub fn matrix_sum(matrix_1: &Matrix, matrix_2: &Matrix) -> Matrix{
        let mut buffer_matrix = Matrix::new(
            matrix_1.rows,
            matrix_2.cols
        );

        for (i, row) in matrix_1.content.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                buffer_matrix.content[i as usize][j as usize] = matrix_1.content[i as usize][j as usize] + matrix_2.content[i as usize][j as usize];
            }
        }

        buffer_matrix
    }

    pub fn matrix_rand(mut self, min: f32, max: f32) -> Self{
        let mut rng = rand::thread_rng();

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.content[i as usize][j as usize] = rng.gen_range(min..max);
            }
        }

        self
    }

    pub fn matrix_sigmoid(mut self) -> Self {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.content[i as usize][j as usize] = sigmoid(self.content[i as usize][j as usize]);
            }
        }

        self
    }

    // ohh my fucking god i'm a bad programmer XD
    pub fn matrix_row(m: &Matrix, n: i32) -> Matrix {
        let mut buffer_matrix = Matrix::new(
            1,
            m.cols
        );

        let mut j= 0;

        for i in &m.content[n as usize] {
            buffer_matrix.content[0][j] = *i;
            j+=1;
        }

        buffer_matrix
    }

    pub fn matrix_copy(mut m1: Matrix, m2: Matrix) -> Matrix {
        assert_eq!(m1.rows, m2.rows);
        assert_eq!(m1.cols, m2.cols);

        for i in 0..m1.rows {
            for j in 0..m1.cols {
                m1.content[i as usize][j as usize] = m2.content[i as usize][j as usize];
            }
        }

        m1
    }

    // copy's the column n2 of m2 into column n1 of m1
    pub fn move_matrix_col(m1: &Matrix, m2: &Matrix, n1: i32, n2: i32) -> Matrix{
        assert_eq!(m1.rows, m2.rows);

        let mut buffer_matrix = Matrix::new(
            m1.rows,
            m1.cols
        );

        for i in 0..m1.rows {
            buffer_matrix.content[i as usize][n1 as usize] = m2.content[i as usize][n2 as usize];
        }

        buffer_matrix
    }
}
