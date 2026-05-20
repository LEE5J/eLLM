use crate::bfloat16::Bf16;
use std::f16;

pub trait Exp {
    fn exp(self) -> Self;
}

impl Exp for f16 {
    fn exp(self) -> Self {
        f16::exp(self)
    }
}


impl Exp for Bf16 {
    fn exp(self) -> Self {
        Bf16::from_f32(self.to_f32().exp())
    }
}

impl Exp for f32 {
    fn exp(self) -> Self {
        f32::exp(self)
    }
}

impl Exp for f64 {
    fn exp(self) -> Self {
        f64::exp(self)
    }
}
