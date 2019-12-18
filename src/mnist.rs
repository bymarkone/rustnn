use log::{info};
use ndarray::prelude::*;
use std::io::{Read, Result};
use std::fs::File;
use byteorder::{ReadBytesExt, BigEndian};

pub fn load(train_size: usize, test_size: usize) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
  return (
    load_images("./data/train-images-idx3-ubyte", train_size).unwrap(), 
    load_labels("./data/train-labels-idx1-ubyte", train_size).unwrap(),
    load_images("./data/t10k-images-idx3-ubyte", test_size).unwrap(), 
    load_labels("./data/t10k-labels-idx1-ubyte", test_size).unwrap(),
  );
}

fn load_labels(filename: &str, size: usize) -> Result<Array1<f32>> {
  let mut file = File::open(filename)?;
  
  let magic_nr = file.read_u32::<BigEndian>()?;
  assert!(magic_nr == 0x0801);

  let nelems = file.read_u32::<BigEndian>()? as usize;
  let nelems = size;
  assert!(nelems > 0);

  let labels = Array::from_shape_vec(nelems, file.bytes().take(size).map(Result::unwrap).map(|x| x as f32).collect()).unwrap();

  assert!(labels.len() == size);
  info!("Found {} labels", labels.len());

  return Ok(labels);
}

fn load_images(filename: &str, size: usize) -> Result<Array2<f32>> {
  let mut file = File::open(filename)?;
  
  let magic_nr = file.read_u32::<BigEndian>()?;
  assert!(magic_nr == 0x0803);

  let nimages = file.read_u32::<BigEndian>()? as usize;
  let nimages = size;
  assert!(nimages > 0);
  
  let nrows = file.read_u32::<BigEndian>()? as usize;
  assert!(nrows == 28);

  let ncols = file.read_u32::<BigEndian>()? as usize;
  assert!(ncols == 28);

  let mut buf = Vec::with_capacity(nimages * nrows * ncols);
  unsafe {
    buf.set_len(nimages * nrows * ncols);
  }

  file.read_exact(&mut buf)?;
  
  let images = Array2::from_shape_vec((nimages, nrows * ncols), buf.to_vec().into_iter().map(|x| x as f32).collect()).unwrap();

  assert!(images.dim() == (nimages, nrows * ncols));
  info!("Found {:?} images", images.dim());
  
  return Ok(images);
}
