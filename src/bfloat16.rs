use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Bf16(u16);

impl Bf16 {
    pub const ZERO: Self = Self(0);
    pub const NEG_INFINITY: Self = Self(0xff80);

    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline(always)]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounding_bias = 0x7fff + lsb;
        Self(((bits.wrapping_add(rounding_bias)) >> 16) as u16)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

impl fmt::Debug for Bf16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bf16({})", self.to_f32())
    }
}

impl PartialOrd for Bf16 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl Add for Bf16 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddAssign for Bf16 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Bf16 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul for Bf16 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div for Bf16 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl Neg for Bf16 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(self.0 ^ 0x8000)
    }
}

impl From<f32> for Bf16 {
    #[inline(always)]
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<Bf16> for f32 {
    #[inline(always)]
    fn from(value: Bf16) -> Self {
        value.to_f32()
    }
}
