#![allow(dead_code)]
#![allow(unused_parens)]
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::borrow::BorrowMut;

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

#[inline(always)]
fn ceil_div_power_of_2(n: u64, p: u64) -> u64 {
    debug_assert!(p.is_power_of_two());
    ((n + 7) & !(p-1)) / p
}

pub struct BitChunks<'a, T: BitChunkAccessor> {
    buffer: &'a [u8],
    accessor: PhantomData<T>,
    bit_offset: usize,
    raw_data: *const u8,
    chunk_len: usize,
    remainder_len: usize,
}

pub struct BitChunkIterator<'a, T: BitChunkAccessor> {
    buffer: &'a [u8],
    accessor: PhantomData<T>,
    bit_offset: usize,
    raw_data: *const u8,
    chunk_len: usize,
    index: usize,
}

pub fn bit_chunk_iterator<T: BitChunkAccessor>(buffer: &[u8], bit_offset: usize) -> BitChunks<'_, T> {
    let bytes = T::bytes();
    let bits = T::bits();
    debug_assert!(bits.is_power_of_two() && bits >= 8 && bits <= 64);

    let byte_offset = (bit_offset / 8);
    let bit_offset = bit_offset % 8;

    let raw_data = unsafe { buffer.as_ptr().offset(byte_offset as isize) };

    let len_bits = (buffer.len()-byte_offset)*8 - bit_offset ;
    let chunk_len = (len_bits) / bits;

    let remainder_len = (len_bits & (bits-1));

    BitChunks::<T> {
        buffer,
        accessor: PhantomData::<T>::default(),
        bit_offset,
        raw_data,
        chunk_len,
        remainder_len,
    }
}

impl <'a, T: BitChunkAccessor> BitChunks<'a, T> {
    pub fn remainder_len(&self) -> usize {
        self.remainder_len
    }

    pub fn remainder_bits(&self) -> u64 {
        let bit_len = self.remainder_len;
        if bit_len == 0 {
            0
        } else {
            let byte_len = ceil_div_power_of_2(bit_len as u64, 8) as usize;

            let mut res = 0_u64;
            for i in 0..byte_len {
                res |= (self.buffer[self.chunk_len * T::bytes() + i] as u64) << (i*8);
            }

            let offset = self.bit_offset as u64;
            if offset != 0 {
                (res >> offset) & !(1 << (64 - offset) - 1)
            } else {
                res
            }
        }
    }

    pub fn remainder_bytes(&self) -> Vec<u8> {
        let bit_len = self.remainder_len;
        if bit_len == 0 {
            vec![]
        } else {
            let bits = self.remainder_bits();

            let byte_len = ceil_div_power_of_2(bit_len as u64, 8) as usize;

            let mut res: Vec<u8> = Vec::with_capacity(byte_len);

            for i in 0..byte_len {
                res.push((bits >> (i*8) & 0xFF) as u8);
            }

            res
        }
    }

    pub fn iter(&self) -> BitChunkIterator<'a, T> {
        BitChunkIterator::<'a, T> {
            buffer: self.buffer,
            accessor: PhantomData::default(),
            bit_offset: self.bit_offset,
            raw_data: self.raw_data,
            chunk_len: self.chunk_len,
            index: 0
        }
    }
}

impl <'a, T: BitChunkAccessor> IntoIterator for BitChunks<'a, T> {
    type Item = u64;
    type IntoIter = BitChunkIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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

    let bitchunks = bit_chunk_iterator::<u64>(valid, offset);

    bitchunks
        .iter()
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

    let remainder_len = bitchunks.remainder_len();
    let remainder_bits = bitchunks.remainder_bits();

    for i in 0..remainder_len {
        if remainder_bits & (1<<i) != 0 {
            sum += remainder[i];
        }
    }

    sum
}

pub fn combine_bitmap(left: &[u8], left_offset: usize, right: &[u8], right_offset: usize, output: &mut [u8]) {
    let chunk_size = <u64 as BitChunkAccessor>::bytes();
    let left_chunks = bit_chunk_iterator::<u64>(left, left_offset);
    let right_chunks = bit_chunk_iterator::<u64>(right, right_offset);
    let mut output_chunks = output.chunks_exact_mut(chunk_size);
    output_chunks
        .borrow_mut()
        .zip(left_chunks.iter().zip(right_chunks.iter()))
        .for_each(|(out, (l, r))| {
        let out: &mut [u64] = unsafe {std::mem::transmute(out) };
        out[0] = l&r;

        //unsafe { (out.as_mut_ptr() as *mut u64).write(l&r) };
    });
    output_chunks.into_remainder()
        .iter_mut()
        .zip(left_chunks.remainder_bytes().iter().zip(right_chunks.remainder_bytes().iter()))
        .for_each(|(out, (l, r))| {
           *out = l&r;
        });

}

#[cfg(test)]
mod tests {
    use crate::{bit_chunk_iterator, ceil_div_power_of_2, aggregate_sum_kernel};

    #[test]
    fn test_ceil() {
        assert_eq!(0, ceil_div_power_of_2(0, 8));
        assert_eq!(1, ceil_div_power_of_2(1, 8));
        assert_eq!(1, ceil_div_power_of_2(7, 8));
        assert_eq!(1, ceil_div_power_of_2(8, 8));
        assert_eq!(2, ceil_div_power_of_2(9, 8));
    }

    #[test]
    fn test_iter_aligned_8() {
        let input: &[u8] = &[0,1,2,4];

        let bitchunks = bit_chunk_iterator::<u8>(input, 0);
        let result = bitchunks.into_iter().collect::<Vec<u64>>();

        assert_eq!(vec![0,1,2,4], result);
    }

    #[test]
    fn test_iter_unaligned_8() {
        let input: &[u8] = &[0b0000000,0b00010001,0b00100010,0b01000100];

        let bitchunks = bit_chunk_iterator::<u8>(input, 1);

        assert_eq!(7, bitchunks.remainder_len());
        assert_eq!(0b00100010, bitchunks.remainder_bits());

        let result = bitchunks.into_iter().collect::<Vec<u64>>();

        assert_eq!(vec![0b10000000, 0b00001000, 0b00010001], result);
    }

    #[test]
    fn test_iter_unaligned_16() {
        let input: &[u8] = &[0b01010101,0b11111111,0b01010101,0b11111111];

        let bitchunks = bit_chunk_iterator::<u16>(input, 1);

        let result = bitchunks.iter().collect::<Vec<u64>>();

        assert_eq!(vec![0b1111111110101010], result);

        assert_eq!(15, bitchunks.remainder_len());
        assert_eq!(0b0111111110101010, bitchunks.remainder_bits());
    }

    #[test]
    fn test_iter_aligned_16() {
        let input: &[u8] = &[0,1,2,4];

        let result = bit_chunk_iterator::<u16>(input, 0).into_iter().collect::<Vec<u64>>();

        assert_eq!(vec![0x0100,0x0402], result);
    }

    #[test]
    fn test_aggregate_sum_kernel() {
        let len = 1000;
        let input: Vec<f32> = (0..len).map(|i| if i % 2 == 0  {2.0} else {1.0}).collect();
        let valid : Vec<u8> = (0..ceil_div_power_of_2(len, 8)).map(|i| 0b01010101).collect();

        let expected: f32 = (0..len).map(|i| if i % 2 == 0  {2.0} else {0.0}).sum();

        let result = aggregate_sum_kernel(&input, &valid, 0);

        assert_eq!(expected, result);
    }

    #[test]
    fn test_combine_bitmap() {
        let left: Vec<f32> = (0..1024).map(|i| if i % 2 == 0  {2.0} else {1.0}).collect();
        let right: Vec<f32> = (0..1024).map(|i| if i % 2 == 0  {2.0} else {1.0}).collect();
    }

}
