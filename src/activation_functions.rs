use std::f32::consts::E;

// sigmoid function
pub fn sigmoid(x: f32) -> f32{
    1. / (1. + E.powf(-x))
}