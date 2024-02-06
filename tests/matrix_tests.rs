use rusty_brain::Matrix;

#[cfg(test)]
mod matrix_tests {
    use super::*;

    #[test]
    fn test_matrix_new() {
        let matrix = Matrix::new(3, 4);
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 4);
        assert_eq!(matrix.content, vec![vec![0.0; 4]; 3]);
    }

    #[test]
    fn test_matrix_dot() {
        let matrix_1 = Matrix {
            rows: 2,
            cols: 3,
            content: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        };

        let matrix_2 = Matrix {
            rows: 3,
            cols: 2,
            content: vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]],
        };

        let result = Matrix::matrix_dot(matrix_1.clone(), matrix_2.clone());

        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(
            result.content,
            vec![vec![58.0, 64.0], vec![139.0, 154.0]]
        );
    }

    #[test]
    fn test_matrix_sum() {
        let matrix_1 = Matrix {
            rows: 2,
            cols: 2,
            content: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };

        let matrix_2 = Matrix {
            rows: 2,
            cols: 2,
            content: vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        };

        let result = Matrix::matrix_sum(matrix_1.clone(), matrix_2.clone());

        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(result.content, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    }
}