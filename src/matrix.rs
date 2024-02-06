#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: i32,
    pub cols: i32,
    pub content: Vec<Vec<f64>>
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

    pub fn print_matrix(matrix: Matrix) {
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                print!("{:?} ", matrix.content[i as usize][j as usize])
            }
            print!("\n");
        }
    }

    pub fn matrix_dot(matrix_1: Matrix, matrix_2: Matrix) -> Matrix{
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

    pub fn matrix_sum(matrix_1: Matrix, matrix_2: Matrix) -> Matrix{
        assert_eq!(matrix_1.rows, matrix_2.rows, "Matrix dimensions mismatch");
        assert_eq!(matrix_1.cols, matrix_2.cols, "Matrix dimensions mismatch");

        let mut buffer_matrix = Matrix::new(
            matrix_1.rows,
            matrix_2.cols
        );

        for i in 0..matrix_1.rows {
            for j in 0..matrix_2.cols {
                buffer_matrix.content[i as usize][j as usize] = matrix_1.content[i as usize][j as usize] + matrix_2.content[i as usize][j as usize];
            }
        }

        buffer_matrix
    }
}
