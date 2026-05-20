use std::ops::{Add, Div, Mul};
use crate::bfloat16::Bf16;
use crate::kernel::generic::from_usize::FromUsize;

pub trait Sqrt:
    Copy + Add<Output = Self> + Mul<Output = Self> + Div<Output = Self> + PartialOrd + FromUsize
{
    fn sqrt(self) -> Self;
}

impl Sqrt for f16 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

impl Sqrt for Bf16 {
    fn sqrt(self) -> Self {
        Bf16::from_f32(self.to_f32().sqrt())
    }
}

impl Sqrt for f32 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

impl Sqrt for f64 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
