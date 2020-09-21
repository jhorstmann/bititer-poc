#![allow(dead_code)]
#![allow(unused_parens)]
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

pub trait BitChunkAccessor: Debug + Display + Copy {

    fn bytes() -> usize {
        return std::mem::size_of::<Self>()
    }

    fn bits() -> usize {
        Self::bytes() * 8
    }

    unsafe fn read(ptr: *const u8, offset: isize) -> u64;
}

impl BitChunkAccessor for u8 {
    unsafe fn read(ptr: *const u8, offset: isize) -> u64 {
        // no need for unaligned read for bytes
        std::ptr::read(ptr.offset(offset)) as u64
    }
}

impl BitChunkAccessor for u16 {
    unsafe fn read(ptr: *const u8, offset: isize) -> u64 {
        std::ptr::read_unaligned((ptr as *const u16).offset(offset)) as u64
    }
}

impl BitChunkAccessor for u32 {
    unsafe fn read(ptr: *const u8, offset: isize) -> u64 {
        std::ptr::read_unaligned((ptr as *const u32).offset(offset)) as u64
    }
}

impl BitChunkAccessor for u64 {
    unsafe fn read(ptr: *const u8, offset: isize) -> u64 {
        std::ptr::read_unaligned((ptr as *const u64).offset(offset))
    }
}



pub struct BitChunkIterator<'a, T: BitChunkAccessor> {
    buffer: &'a [u8],
    accessor: PhantomData<T>,
    bit_offset: usize,
    raw_data: *const u8,
    chunk_len: usize,
    remainder_len: usize,
    index: usize,
}

#[inline(always)]
fn ceil_power_of_2(n: u64, p: u64) -> u64 {
    debug_assert!(p.is_power_of_two());
    (n + 7) & !(p-1)
}

#[inline(always)]
fn floor_power_of_2(n: usize, p: usize) -> usize {
    debug_assert!(p.is_power_of_two());
    (n) & !(p-1)
}

pub fn bit_chunk_iterator<T: BitChunkAccessor>(buffer: &[u8], bit_offset: usize) -> BitChunkIterator<'_, T> {
    let bytes = T::bytes();
    let bits = T::bits();
    debug_assert!(bits.is_power_of_two() && bits >= 8 && bits <= 64);

    let byte_offset = (bit_offset / 8);
    let bit_offset = bit_offset % 8;

    let raw_data = unsafe { buffer.as_ptr().offset(byte_offset as isize) };

    let len_bits = (buffer.len()-byte_offset)*8 - bit_offset ;
    let chunk_len = (len_bits) / bits;

    dbg!(buffer.len());
    dbg!(len_bits);

    let remainder_len = (len_bits & (bits-1));

    BitChunkIterator::<T> {
        buffer,
        accessor: PhantomData::<T>::default(),
        bit_offset,
        raw_data,
        chunk_len,
        remainder_len,
        index: 0,
    }
}

impl <T: BitChunkAccessor> BitChunkIterator<'_, T> {
    fn remainder_len(&self) -> usize {
        self.remainder_len
    }

    fn remainder_bits(&self) -> u64 {
        if self.remainder_len == 0 {
            0
        } else {
            let mut res = 0_u64;
            for i in 0..ceil_power_of_2(self.remainder_len as u64, 8) as usize / 8 {
                res |= (self.buffer[self.chunk_len * T::bytes() + i] as u64) << (i*8);
            }

            let offset = self.bit_offset as u64;
            if (offset != 0) {
                (res >> offset) & !(1 << (64 - offset) - 1)
            } else {
                res
            }
        }
    }
}

impl <T: BitChunkAccessor> Iterator for BitChunkIterator<'_, T> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.chunk_len {
            return None;
        }

        let current = unsafe { T::read(self.raw_data, self.index as isize) };

        let combined = if self.bit_offset == 0 {
            current
        } else {
            let next = unsafe { T::read(self.raw_data, self.index as isize + 1) };
            current >> self.bit_offset | (next & ((1 << self.bit_offset) - 1)) << (T::bits() - self.bit_offset)
        };

        self.index += 1;

        Some(combined)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.chunk_len - self.index, Some(self.chunk_len - self.index))
    }
}

pub fn aggregate_sum_kernel(input: &[f32], valid: &[u8], offset: usize) -> f32 {
    let chunks = input[offset..].chunks_exact(64);
    let remainder = chunks.remainder();

    let sum = &mut [0_f32; 64];

    let bititer = bit_chunk_iterator::<u64>(valid, offset);

    let remainder_len = bititer.remainder_len();
    let remainder_bits = bititer.remainder_bits();

    bititer
        .zip(chunks.into_iter())
        .for_each(|(mask, slice)| {
            for i in 0..64 {
                let blend = if (mask & (1<<i)) != 0 {
                    1.0
                } else {
                    0.0
                };
                sum[i] += blend * (slice[i]);
            }
        });

    let mut sum: f32 = sum.iter().sum();

    for i in 0..remainder_len {
        if remainder_bits & (1<<i) != 0 {
            sum += remainder[i];
        }
    }

    sum
}


pub fn mul_kernel(left: &[f32], right: &[f32], valid: &[u8], offset: usize, output: &mut[f32]) {
    let mask_size = <u64 as BitChunkAccessor>::bits();
    let output_chunks = output.chunks_exact_mut(mask_size);
    let left_chunks = left[offset..].chunks_exact(mask_size);
    let right_chunks = right[offset..].chunks_exact(mask_size);

    bit_chunk_iterator::<u64>(valid, offset)
        .zip(output_chunks.into_iter()
                 .zip(left_chunks.into_iter().zip(right_chunks.into_iter())))
        .for_each(|(mask, (out, (l, r)))| {
            for i in 0..mask_size {
                let blend = if (mask & (1<<i)) != 0 {
                    1.0
                } else {
                    0.0
                };
                out[i] = blend * (l[i] * r[i]);
            }
        });
}


pub fn cmp_kernel(left: &[f32], right: &[f32], valid: &[u8], offset: usize, output: &mut[u8]) {
    let chunk_size = <u64 as BitChunkAccessor>::bits();
    let output_chunks = output.chunks_exact_mut(chunk_size);
    let left_chunks = left[offset..].chunks_exact(chunk_size);
    let right_chunks = right[offset..].chunks_exact(chunk_size);

    bit_chunk_iterator::<u64>(valid, offset)
        .zip(output_chunks.into_iter()
            .zip(left_chunks.into_iter().zip(right_chunks.into_iter())))
        .for_each(|(mask, (out, (l, r)))| {
            let out: &mut [u64] = unsafe {std::mem::transmute(out) };
            let mut res_mask = 0_u64;
            for i in 0..chunk_size {
                let bit = if (mask & (1<<i)) != 0 {
                    1
                } else {
                    0
                };
                res_mask |= if (l[i] == r[i]) {
                    1 << i
                } else {
                    0
                };
            }
            out[0] = res_mask;
        });
}

pub fn combine_bitmap(left: &[u8], left_offset: usize, right: &[u8], right_offset: usize, output: &mut[u8]) {
    let chunk_size = <u64 as BitChunkAccessor>::bytes();
    output.chunks_exact_mut(chunk_size).zip(
        bit_chunk_iterator::<u64>(left, left_offset)
            .zip(bit_chunk_iterator::<u64>(right, right_offset))
    ).for_each(|(out, (l, r))| {
        let out: &mut [u64] = unsafe {std::mem::transmute(out) };
        out[0] = l&r;

        //unsafe { (out.as_mut_ptr() as *mut u64).write(l&r) };
    })

}

#[cfg(test)]
mod tests {
    use crate::{bit_chunk_iterator, ceil_power_of_2, floor_power_of_2, mul_kernel, aggregate_sum_kernel};

    #[test]
    fn test_ceil() {
        assert_eq!(0, ceil_power_of_2(0, 8));
        assert_eq!(8, ceil_power_of_2(1, 8));
        assert_eq!(8, ceil_power_of_2(7, 8));
        assert_eq!(8, ceil_power_of_2(8, 8));
        assert_eq!(16, ceil_power_of_2(9, 8));
    }

    #[test]
    fn test_floor_bits() {
        assert_eq!(0, floor_power_of_2(0, 8));
        assert_eq!(0, floor_power_of_2(1, 8));
        assert_eq!(0, floor_power_of_2(7, 8));
        assert_eq!(8, floor_power_of_2(8, 8));
        assert_eq!(8, floor_power_of_2(9, 8));
    }

    #[test]
    fn test_iter_aligned_8() {
        let input: &[u8] = &[0,1,2,4];

        let result = bit_chunk_iterator::<u8>(input, 0).collect::<Vec<u64>>();

        assert_eq!(vec![0,1,2,4], result);
    }

    #[test]
    fn test_iter_unaligned_8() {
        let input: &[u8] = &[0b0000000,0b00010001,0b00100010,0b01000100];

        let bititer = bit_chunk_iterator::<u8>(input, 1);

        assert_eq!(7, bititer.remainder_len());
        assert_eq!(0b00100010, bititer.remainder_bits());

        let result = bititer.collect::<Vec<u64>>();

        assert_eq!(vec![0b10000000, 0b00001000, 0b00010001], result);
    }

    #[test]
    fn test_iter_unaligned_16() {
        let input: &[u8] = &[0b01010101,0b11111111,0b01010101,0b11111111];

        let bititer = bit_chunk_iterator::<u16>(input, 1);

        assert_eq!(15, bititer.remainder_len());
        assert_eq!(0b0111111110101010, bititer.remainder_bits());

        let result = bititer.collect::<Vec<u64>>();

        assert_eq!(vec![0b1111111110101010], result);
    }

    #[test]
    fn test_iter_aligned_16() {
        let input: &[u8] = &[0,1,2,4];

        let result = bit_chunk_iterator::<u16>(input, 0).collect::<Vec<u64>>();

        assert_eq!(vec![0x0100,0x0402], result);
    }

    #[test]
    fn test_mul_kernel() {
        let left: Vec<f32> = (0..1024).map(|i| if i % 2 == 0  {2.0} else {1.0}).collect();
        let right: Vec<f32> = (0..1024).map(|i| if i % 2 == 0  {2.0} else {1.0}).collect();
        let mut out = Vec::<f32>::with_capacity(1024);
        unsafe {out.set_len(1024)};
        let valid : Vec<u8> = (0..1024/8).map(|i| 0b01010101).collect();

        let expected: Vec<f32> = (0..1024).map(|i| if i % 2 == 0  {4.0} else {0.0}).collect();

        mul_kernel(&left, &right, &valid, 0, &mut out);

        assert_eq!(expected, out);

    }

    #[test]
    fn test_aggregate_sum_kernel() {
        let len = 1024;
        let input: Vec<f32> = (0..len).map(|i| if i % 2 == 0  {2.0} else {1.0}).collect();
        let valid : Vec<u8> = (0..ceil_power_of_2(len, 8)/8).map(|i| 0b01010101).collect();

        let expected: f32 = (0..len).map(|i| if i % 2 == 0  {2.0} else {0.0}).sum();

        let result = aggregate_sum_kernel(&input, &valid, 0);

        assert_eq!(expected, result);

    }

}
